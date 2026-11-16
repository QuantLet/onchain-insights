"""Quantile forecasts (24 horizons) for the decision backtest.

Forecast artifacts are reduced by extract_quantiles.py to 26 quantile levels (pairs bracketing
0.5%, 1%, 2.5%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 97.5%, 99%, 99.5% on the model u-grid).
Origin i corresponds to panel row ORIGIN0 + i (2025-01-15 14:00 UTC); the h-step target of
origin i is the depeg of bucket ORIGIN0 + i + h (verified against the stored `ds`).
"""
import numpy as np
import pandas as pd
from data import panel

U_SEL = np.array([0.004599481821060181, 0.005929410457611084, 0.00996282696723938, 0.011322170495986938,
                  0.023911774158477783, 0.025352299213409424, 0.04969799518585205, 0.05130910873413086,
                  0.09804826974868774, 0.10007324814796448, 0.24742251634597778, 0.2526317834854126,
                  0.4212549328804016, 0.5787450671195984, 0.7473682165145874, 0.7525774836540222,
                  0.8999267816543579, 0.9019517302513123, 0.9486908912658691, 0.950302004814148,
                  0.9746477007865906, 0.9760882258415222, 0.9886778593063354, 0.990037202835083,
                  0.9940705895423889, 0.9954005479812622])
ORIGIN0_TS = pd.Timestamp('2025-01-15 14:00', tz='UTC')


class QForecast:
    def __init__(self, Q, name, origin_ts):
        self.Q, self.name, self.u, self.origin_ts = Q, name, U_SEL, origin_ts

    @classmethod
    def from_npz(cls, path, name):
        z = np.load(path)
        Q = z['Qsel']
        P = panel()
        r0 = P.index.get_loc(ORIGIN0_TS)
        origin_ts = P.index[r0:r0 + Q.shape[0]]
        if 'ds' in z.files:
            ds = pd.to_datetime(z['ds'][:, 0], unit='s', utc=True)
            ok = float((ds == P.index[r0 + 1:r0 + 1 + Q.shape[0]]).mean())
            if ok < 0.999:
                raise ValueError(f'{name}: forecast timestamps do not align with the panel ({ok:.3f})')
        return cls(Q, name, origin_ts)

    @classmethod
    def rolling_empirical(cls, window=24 * 30, name='Rolling empirical'):
        """No-ML benchmark: d_{t+h} = d_t + empirical quantiles of h-step changes, trailing 30 days."""
        P = panel(); d = P['depeg_bps'].values
        r0 = P.index.get_loc(ORIGIN0_TS); n = len(P) - r0 - 25
        Q = np.zeros((n, 24, len(U_SEL)))
        for i in range(n):
            r = r0 + i
            for h in range(1, 25):
                ch = d[r - window + h:r + 1] - d[r - window:r + 1 - h]
                Q[i, h - 1] = d[r] + np.quantile(ch, U_SEL)
        return cls(Q, name, P.index[r0:r0 + n])

    def q(self, a):
        j = int(np.clip(np.searchsorted(self.u, a), 1, len(self.u) - 1))
        u0, u1 = self.u[j - 1], self.u[j]
        w = float(np.clip((a - u0) / (u1 - u0), 0, 1))
        return self.Q[:, :, j - 1] * (1 - w) + self.Q[:, :, j] * w

    def interval(self, m, alpha, H, lag=1):
        """Interval (bps) used at decision k (end of bucket k): origin stamped k-lag (1h data lag),
        horizons covering buckets k+1..k+H. alpha=1 returns the min/max of the medians."""
        lo_all = self.q(alpha / 2); hi_all = self.q(1 - alpha / 2)
        hs = np.arange(lag, min(lag + H, 24))
        lo = pd.Series(lo_all[:, hs].min(1), index=self.origin_ts)
        hi = pd.Series(hi_all[:, hs].max(1), index=self.origin_ts)
        st = m.idx - pd.Timedelta(hours=lag)
        return lo.reindex(st).ffill().bfill().values, hi.reindex(st).ffill().bfill().values

    def median_path(self, m, H, lag=1):
        lo, hi = self.interval(m, 1.0, H, lag)
        return (lo + hi) / 2
