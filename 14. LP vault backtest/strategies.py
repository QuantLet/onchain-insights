"""Vault strategies. decide(k, P, pos) is called at the END of hour k (k = -1 for the initial placement) and
returns ('keep',) or ('lp', tick_lower, tick_upper). pos = dict(L, ta, tb) describes the current position."""
import numpy as np
from lpsim import tick_of


# ----------------------------------------------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------------------------------------------
def episodes_from_flags(raw, quiet=12):
    """Alert-episode state with a quiet-period reset: active from the first raised flag until the flag has been
    off for `quiet` consecutive hours."""
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


def depeg_events(depeg, thr=15.0, reset=24):
    """Indices of first breaches of |d| >= thr, declustered with a `reset`-hour reset."""
    ev, last = [], -10**9
    for i, b in enumerate(np.abs(depeg) >= thr):
        if b:
            if i - last > reset:
                ev.append(i)
            last = i
    return np.array(ev, int)


def _out(P, pos):
    return P < 1.0001 ** pos['ta'] or P > 1.0001 ** pos['tb']


# ----------------------------------------------------------------------------------------------------------
# strategies
# ----------------------------------------------------------------------------------------------------------
class Strategy:
    name, mode = 'base', 0

    def reset(self, m):
        self.m, self.mode = m, 0


class Static(Strategy):
    """Fixed range of +-w ticks around parity, never repositioned."""
    def __init__(self, w):
        self.w, self.name = w, f'Static ±{w}'

    def decide(self, k, P, pos):
        return ('lp', -self.w, self.w) if k == -1 else ('keep',)


class Recenter(Strategy):
    """Gamma-style vault: +-w ticks around the current tick, recentred whenever the price leaves the range."""
    def __init__(self, w):
        self.w, self.name = w, f'Gamma ±{w}'

    def decide(self, k, P, pos):
        if pos['L'] == 0 or _out(P, pos):
            c = tick_of(P)
            return ('lp', c - self.w, c + self.w + 1)
        return ('keep',)


class ForecastInterval(Strategy):
    """Asymmetric forecaster range: the predictive interval [lo_k, hi_k] (bp, lagged, union over the next hours),
    extended to contain the current price. Repositioned when the price leaves the range or when the target width
    becomes more than twice / less than a third of the current width."""
    def __init__(self, lo, hi, name='Forecaster range', wmin=2, wmax=120):
        self.lo, self.hi, self.wmin, self.wmax, self.name = np.asarray(lo), np.asarray(hi), wmin, wmax, name

    def target(self, k, P):
        kk, c = max(k, 0), tick_of(P)
        ta, tb = int(np.floor(min(self.lo[kk], c))), int(np.ceil(max(self.hi[kk], c + 1)))
        if tb - ta < self.wmin:
            tb = ta + self.wmin
        if tb - ta > self.wmax:
            ta, tb = max(ta, c - self.wmax // 2), min(tb, c + 1 + self.wmax // 2)
        return ta, tb

    def decide(self, k, P, pos):
        ta, tb = self.target(k, P)
        if pos['L'] == 0:
            return ('lp', ta, tb)
        cur, new = pos['tb'] - pos['ta'], tb - ta
        if _out(P, pos) or new > 2 * cur or 3 * new < cur:
            return ('lp', ta, tb)
        return ('keep',)


class ForecastSymmetric(Strategy):
    """Symmetric variant (robustness): half-width = half the interval width, centred on the median forecast
    (kept within reach of the current price). Same repositioning triggers as ForecastInterval."""
    def __init__(self, lo, hi, med, name='Forecaster range (symmetric)', wmin=1, wmax=60):
        self.lo, self.hi, self.med = np.asarray(lo), np.asarray(hi), np.asarray(med)
        self.wmin, self.wmax, self.name = wmin, wmax, name

    def target(self, k, P):
        kk, c = max(k, 0), tick_of(P)
        w = int(np.clip(np.ceil((self.hi[kk] - self.lo[kk]) / 2), self.wmin, self.wmax))
        cm = min(max(int(np.floor(self.med[kk])), c - w), c + w)
        return w, cm - w, cm + w + 1

    def decide(self, k, P, pos):
        w, ta, tb = self.target(k, P)
        if pos['L'] == 0:
            return ('lp', ta, tb)
        cur = (pos['tb'] - pos['ta'] - 1) / 2
        if _out(P, pos) or w > 2 * cur + 1 or (2 * w + 1) * 3 < (pos['tb'] - pos['ta']):
            return ('lp', ta, tb)
        return ('keep',)


class Freeze(Strategy):
    """Classifier gate: while alert[k] is on, the vault keeps its current position and does not reposition.
    alert[k] is the alert state usable at decision k (already lagged by the caller)."""
    def __init__(self, base, alert, name=None):
        self.base, self.alert = base, np.asarray(alert, bool)
        self.name = name or f'{base.name} + freeze'

    def reset(self, m):
        super().reset(m); self.base.reset(m)

    def decide(self, k, P, pos):
        a = bool(self.alert[k]) if k >= 0 else False
        self.mode = int(a)
        if a and pos['L'] > 0:
            return ('keep',)
        return self.base.decide(k, P, pos)
