import numpy as np
from lpsim import tick_of

# --------------------------------------------------------------------------
# signal helpers
# --------------------------------------------------------------------------

def episodes_from_flags(raw, quiet=12):
    """Alert-episode state with a quiet-period reset: active from first crossing until
    the raw flag has been off for `quiet` consecutive hours."""
    act = np.zeros(len(raw), bool); off = quiet; on = False
    for i, r in enumerate(raw):
        if r:
            on = True; off = 0
        else:
            off += 1
            if off >= quiet:
                on = False
        act[i] = on
    return act


def episode_onsets(active):
    a = np.asarray(active, bool)
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])


def depeg_events(depeg, idx, thr=15.0, reset=24):
    """First-breach times of |d|>=thr events declustered with a `reset`-hour reset."""
    br = np.abs(depeg) >= thr
    ev = []; last_breach = -10**9
    for i, b in enumerate(br):
        if b:
            if i - last_breach > reset:
                ev.append(i)
            last_breach = i
    return np.array(ev, int)


# --------------------------------------------------------------------------
# strategies: decide(k, P, pos) -> ('keep',) | ('lp', ta, tb) | ('hold', frac|None)
# decide(k, ...) is called at the END of bucket k (k=-1 for initial placement)
# --------------------------------------------------------------------------

class Strategy:
    name = 'base'
    mode = 0

    def reset(self, m):
        self.m = m; self.mode = 0


class Hold(Strategy):
    def __init__(self):
        self.name = 'HODL 50/50'

    def decide(self, k, P, pos):
        return ('keep',)


class Static(Strategy):
    def __init__(self, w):
        self.w = w; self.name = f'Static parity ±{w}'

    def decide(self, k, P, pos):
        if k == -1:
            return ('lp', -self.w, self.w)
        return ('keep',)


def centred(P, w):
    c = tick_of(P)
    return c - w, c + w + 1


class Recenter(Strategy):
    """Gamma-style: narrow range around the current tick, recentred when price exits."""
    def __init__(self, w):
        self.w = w; self.name = f'Recenter ±{w}'

    def decide(self, k, P, pos):
        if pos['L'] == 0:
            return ('lp',) + centred(P, self.w)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        if P < pa or P > pb:
            return ('lp',) + centred(P, self.w)
        return ('keep',)


class VolRecenter(Strategy):
    """Heuristic adaptive width: half-width = mult x trailing 24h range of the depeg (bps)."""
    def __init__(self, mult=1.0, lookback=24, wmin=1, wmax=60):
        self.mult, self.lb, self.wmin, self.wmax = mult, lookback, wmin, wmax
        self.name = f'Vol-scaled recenter (x{mult})'

    def reset(self, m):
        super().reset(m)
        d = m.depeg
        self.width = np.zeros(m.n)
        for k in range(m.n):
            seg = d[max(0, k - self.lb + 1):k + 1]
            self.width[k] = np.clip(np.ceil(self.mult * (seg.max() - seg.min()) / 2), self.wmin, self.wmax)

    def decide(self, k, P, pos):
        w = int(self.width[max(k, 0)])
        if pos['L'] == 0:
            return ('lp',) + centred(P, w)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        cur = (pos['tb'] - pos['ta'] - 1) / 2
        if P < pa or P > pb or w > 2 * cur + 1 or (2 * w + 1) * 3 < (pos['tb'] - pos['ta']):
            return ('lp',) + centred(P, w)
        return ('keep',)


class Gated(Strategy):
    """Wrap a base strategy; in alert mode apply a protective action.
    alert[k] = alert state usable at decision k (already lagged by the caller).
    action: 'freeze' (no rebalancing), 'wide' (parity-anchored ±W, no rebalancing),
            'exit' (burn, hold tokens as they are), 'exit5050' (burn, swap to 50/50)."""
    def __init__(self, base, alert, action='wide', W=30, name=None):
        self.base, self.alert, self.action, self.W = base, np.asarray(alert, bool), action, W
        self.name = name or f'{base.name} | gate:{action}'

    def reset(self, m):
        super().reset(m); self.base.reset(m); self.prev = False

    def decide(self, k, P, pos):
        a = bool(self.alert[k]) if k >= 0 else False
        onset = a and not self.prev
        ended = (not a) and self.prev
        self.prev = a; self.mode = int(a)
        if not a:
            if ended and self.action in ('wide', 'exit', 'exit5050'):
                return self.base.decide(k, P, dict(L=0.0, ta=None, tb=None))
            return self.base.decide(k, P, pos)
        if self.action == 'freeze':
            if pos['L'] == 0:
                return self.base.decide(k, P, pos)
            return ('keep',)
        if self.action == 'wide':
            if onset or pos['L'] == 0:
                return ('lp', -self.W, self.W)
            return ('keep',)
        if self.action == 'exit':
            return ('hold', None) if onset or pos['L'] > 0 else ('keep',)
        if self.action == 'exit5050':
            return ('hold', 0.5) if onset or pos['L'] > 0 else ('keep',)
        raise ValueError(self.action)


class ForecastRange(Strategy):
    """Range from the predictive distribution: [lo_k, hi_k] in bps (already lagged and
    aggregated over horizons by the caller). Rebalance when price exits the range, when
    the forecast interval is no longer covered, or when the range is > `tight` x too wide."""
    def __init__(self, lo, hi, name='Forecast range', tight=3.0, pad=0):
        self.lo, self.hi, self.tight, self.pad = np.asarray(lo), np.asarray(hi), tight, pad
        self.name = name

    def target(self, k, P):
        kk = max(k, 0)
        lo, hi = self.lo[kk], self.hi[kk]
        c = tick_of(P)
        ta = int(np.floor(min(lo, c))) - self.pad
        tb = int(np.ceil(max(hi, c + 1))) + self.pad
        if tb - ta < 2:
            tb = ta + 2
        return ta, tb

    def decide(self, k, P, pos):
        ta, tb = self.target(k, P)
        if pos['L'] == 0:
            return ('lp', ta, tb)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        out = P < pa or P > pb
        uncovered = ta < pos['ta'] or tb > pos['tb']
        too_wide = (pos['tb'] - pos['ta']) > self.tight * (tb - ta)
        if out or uncovered or too_wide:
            return ('lp', ta, tb)
        return ('keep',)


class Combined(Strategy):
    """Classifier gates the regime; forecaster sets the range in both regimes.
    Normal: central forecast interval (base ForecastRange).
    Alert : wide tail interval (e.g. 1%-99% over 24h) placed once at alert onset and
            frozen (no rebalancing) until the alert episode ends."""
    def __init__(self, base, alert, lo_tail, hi_tail, name='Combined'):
        self.base, self.alert = base, np.asarray(alert, bool)
        self.lo_t, self.hi_t = np.asarray(lo_tail), np.asarray(hi_tail)
        self.name = name

    def reset(self, m):
        super().reset(m); self.base.reset(m); self.prev = False

    def decide(self, k, P, pos):
        a = bool(self.alert[k]) if k >= 0 else False
        onset = a and not self.prev; ended = (not a) and self.prev
        self.prev = a; self.mode = int(a)
        if not a:
            if ended:
                return self.base.decide(k, P, dict(L=0.0, ta=None, tb=None))
            return self.base.decide(k, P, pos)
        if onset or pos['L'] == 0:
            kk = max(k, 0); c = tick_of(P)
            ta = int(np.floor(min(self.lo_t[kk], c))); tb = int(np.ceil(max(self.hi_t[kk], c + 1)))
            return ('lp', ta, tb)
        return ('keep',)


class ForecastWidth(Strategy):
    """Like VolRecenter, but the half-width comes from the forecast interval and the centre
    from the forecast median: recentre on exit, or when the target width changes by >2x."""
    def __init__(self, lo, hi, med=None, name='Forecast width', wmin=1, wmax=60, use_median=True):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.med = None if med is None else np.asarray(med)
        self.wmin, self.wmax, self.use_median = wmin, wmax, use_median
        self.name = name

    def tgt(self, k, P):
        kk = max(k, 0)
        w = int(np.clip(np.ceil((self.hi[kk] - self.lo[kk]) / 2), self.wmin, self.wmax))
        c = tick_of(P)
        if self.use_median and self.med is not None:
            cm = int(np.floor(self.med[kk]))
            # keep the current price inside the range
            cm = min(max(cm, c - w), c + w)
            c = cm
        return w, c - w, c + w + 1

    def decide(self, k, P, pos):
        w, ta, tb = self.tgt(k, P)
        if pos['L'] == 0:
            return ('lp', ta, tb)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        cur = (pos['tb'] - pos['ta'] - 1) / 2
        if P < pa or P > pb or w > 2 * cur + 1 or (2 * w + 1) * 3 < (pos['tb'] - pos['ta']):
            return ('lp', ta, tb)
        return ('keep',)


class ForecastRangeH(ForecastRange):
    """ForecastRange with hysteresis: reposition on price exit, or when the forecast
    interval sticks out of the current range by >= `slack` ticks, or when the range is
    > tight x the target; non-exit repositioning at most once every `cool` hours."""
    def __init__(self, lo, hi, name='Forecast range (hyst)', tight=3.0, slack=2, cool=6, pad=0):
        super().__init__(lo, hi, name=name, tight=tight, pad=pad)
        self.slack, self.cool = slack, cool

    def reset(self, m):
        super().reset(m); self.last = -10**9

    def decide(self, k, P, pos):
        ta, tb = self.target(k, P)
        if pos['L'] == 0:
            self.last = k; return ('lp', ta, tb)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        if P < pa or P > pb:
            self.last = k; return ('lp', ta, tb)
        if k - self.last >= self.cool:
            unc = (pos['ta'] - ta) >= self.slack or (tb - pos['tb']) >= self.slack
            too_wide = (pos['tb'] - pos['ta']) > self.tight * (tb - ta)
            if unc or too_wide:
                self.last = k; return ('lp', ta, tb)
        return ('keep',)
