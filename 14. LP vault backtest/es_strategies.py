"""Expected-shortfall (ES) based LP strategies combining the forecaster's tails with the classifier.

ES at level a from a quantile forecast: ES_lo(a) = (1/a) int_0^a q(u) du and ES_hi(a) = (1/a) int_{1-a}^1 q(u) du,
computed on a fine u-grid by linear interpolation of the extracted quantile levels (flat beyond the outermost
level, which understates the tails equally for every forecaster).

Strategies (all decisions at the end of hour k for hour k+1, model inputs from origin k-1):
  ESBand        asymmetric range [med - kappa (med - ES_lo), med + kappa (ES_hi - med)] over the next H hours;
                reposition on price exit or when the target width changes by > 2x (as the width rule).
  ESSized       forecast-width ranges, but only a fraction phi_k = clip(ref_k / tail_k, phi_min, 1) of capital is
                deployed, where tail_k is the 24h ES distance from the median and ref_k its trailing 30-day median;
                resized when phi moves by more than 25%.
  Kumo AND ES   classifier alert episodes kept only while the forecaster's 24h ES distance from the current price
                exceeds tau bp (filters calm-period false alerts); action freeze.
  ESDislocation on classifier alert, an asymmetric range spanning the current price and the 1% ES on each side over
                24h, held until the episode ends (provide liquidity into the dislocation, capture the reversion).
"""
import numpy as np
import pandas as pd

from lpsim import tick_of
from strategies import Strategy, ForecastWidth, Gated, Combined, episodes_from_flags
from forecasts import U_SEL


def _es_weights(a, upper=False, n=400):
    """Linear weights w (len(U_SEL)) such that ES = Q @ w (interpolation is linear in Q)."""
    u = (np.linspace(1 - a, 1, n) if upper else np.linspace(0, a, n))
    W = np.zeros((n, len(U_SEL)))
    for i, uu in enumerate(u):
        if uu <= U_SEL[0]:
            W[i, 0] = 1
        elif uu >= U_SEL[-1]:
            W[i, -1] = 1
        else:
            j = np.searchsorted(U_SEL, uu)
            t = (uu - U_SEL[j - 1]) / (U_SEL[j] - U_SEL[j - 1])
            W[i, j - 1] = 1 - t; W[i, j] = t
    return W.mean(0)


def es_paths(f, a):
    """(N, 24) lower and upper ES of each origin/horizon."""
    return f.Q @ _es_weights(a, False), f.Q @ _es_weights(a, True)


def es_interval(f, m, a, H, lag=1):
    """Lowest lower-ES and highest upper-ES over the horizons covering buckets k+1..k+H; plus the median path."""
    lo, hi = es_paths(f, a)
    hs = np.arange(lag, min(lag + H, 24))
    s_lo = pd.Series(lo[:, hs].min(1), index=f.origin_ts)
    s_hi = pd.Series(hi[:, hs].max(1), index=f.origin_ts)
    st = m.idx - pd.Timedelta(hours=lag)
    return (s_lo.reindex(st).ffill().bfill().values, s_hi.reindex(st).ffill().bfill().values,
            f.median_path(m, H, lag))


class ESBand(Strategy):
    def __init__(self, lo, hi, med, kappa=1.0, name='ES band', wmax=80):
        self.lo, self.hi, self.med, self.kappa, self.wmax = lo, hi, med, kappa, wmax
        self.name = name

    def tgt(self, k, P):
        kk = max(k, 0); c = tick_of(P)
        a = self.med[kk] - self.kappa * (self.med[kk] - self.lo[kk])
        b = self.med[kk] + self.kappa * (self.hi[kk] - self.med[kk])
        ta = int(np.floor(min(a, c))); tb = int(np.ceil(max(b, c + 1)))
        ta = max(ta, c - self.wmax); tb = min(tb, c + 1 + self.wmax)
        if tb - ta < 2:
            tb = ta + 2
        return ta, tb

    def decide(self, k, P, pos):
        ta, tb = self.tgt(k, P)
        if pos['L'] == 0:
            return ('lp', ta, tb)
        pa, pb = 1.0001 ** pos['ta'], 1.0001 ** pos['tb']
        w_cur, w_new = pos['tb'] - pos['ta'], tb - ta
        if P < pa or P > pb or w_new > 2 * w_cur or 3 * w_new < w_cur:
            return ('lp', ta, tb)
        return ('keep',)


class ESSized(Strategy):
    """Wraps a range strategy; scales the deployed share of capital by the forecast tail."""
    def __init__(self, base, phi, name='ES-sized'):
        self.base, self.phi, self.name = base, np.asarray(phi), name

    def reset(self, m):
        super().reset(m); self.base.reset(m); self.cur = None

    def decide(self, k, P, pos):
        phi = float(self.phi[max(k, 0)])
        a = self.base.decide(k, P, pos)
        if a[0] == 'lp':
            self.cur = phi
            return ('lp', a[1], a[2], phi)
        if pos['L'] > 0 and self.cur is not None and (phi > 1.25 * self.cur or phi < self.cur / 1.25):
            self.cur = phi
            return ('lp', pos['ta'], pos['tb'], phi)
        return a


def es_phi(f, m, a=0.05, H=24, phi_min=0.25, lag=1):
    lo, hi, med = es_interval(f, m, a, H, lag)
    tail = np.maximum(med - lo, hi - med)
    s = pd.Series(tail, index=m.idx)
    ref = s.rolling(24 * 30, min_periods=24).median().bfill()
    return np.clip(ref.values / np.maximum(tail, 1e-9), phi_min, 1.0)


def es_tail_distance(f, m, a=0.05, H=24, lag=1):
    """Largest distance (bp) between the current depeg and the lower/upper ES over the next H hours."""
    lo, hi, _ = es_interval(f, m, a, H, lag)
    d = m.depeg
    return np.maximum(d - lo, hi - d)
