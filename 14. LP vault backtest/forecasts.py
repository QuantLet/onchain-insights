"""Quantile forecasts: origins x 24 horizons x 26 quantile levels (see extract_quantiles.py)."""
import numpy as np
import pandas as pd
import config as C
from data import panel

U_SEL = np.array([0.004599481821060181, 0.005929410457611084, 0.00996282696723938, 0.011322170495986938,
                  0.023911774158477783, 0.025352299213409424, 0.04969799518585205, 0.05130910873413086,
                  0.09804826974868774, 0.10007324814796448, 0.24742251634597778, 0.2526317834854126,
                  0.4212549328804016, 0.5787450671195984, 0.7473682165145874, 0.7525774836540222,
                  0.8999267816543579, 0.9019517302513123, 0.9486908912658691, 0.950302004814148,
                  0.9746477007865906, 0.9760882258415222, 0.9886778593063354, 0.990037202835083,
                  0.9940705895423889, 0.9954005479812622])


class QForecast:
    def __init__(self, Q, name, origin_ts):
        self.Q, self.name, self.origin_ts = Q, name, pd.DatetimeIndex(origin_ts)

    @classmethod
    def from_npz(cls, path, name):
        z = np.load(path)
        Q = z['Qsel']
        if 'ds' in z.files:        # target time of horizon 1 = origin + 1h
            origin = pd.to_datetime(z['ds'][:, 0], unit='s', utc=True) - pd.Timedelta(hours=1)
        else:
            r0 = panel().index.get_loc(pd.Timestamp(C.FIRST_ORIGIN, tz='UTC'))
            origin = panel().index[r0:r0 + Q.shape[0]]
        return cls(Q, name, origin)

    @classmethod
    def rolling_empirical(cls, start, end, window=24 * 30):
        """No-ML benchmark: d_{t+h} = d_t + empirical quantiles of h-hour changes over the trailing 30 days."""
        cache = C.HERE / f'.rolling_empirical_{pd.Timestamp(start):%Y%m%d%H}_{pd.Timestamp(end):%Y%m%d%H}.npz'
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if cache.exists():
            z = np.load(cache)
            return cls(z['Q'], 'Rolling empirical', pd.to_datetime(z['ts'], unit='s', utc=True))
        P = panel(); d = P['depeg_bps'].values
        ts_ = lambda t: pd.Timestamp(t).tz_localize('UTC') if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)
        r0 = P.index.get_loc(ts_(start)); r1 = min(P.index.get_loc(ts_(end)), len(P) - 25)
        Q = np.zeros((r1 - r0, 24, len(U_SEL)))
        for i in range(r1 - r0):
            r = r0 + i
            for h in range(1, 25):
                Q[i, h - 1] = d[r] + np.quantile(d[r - window + h:r + 1] - d[r - window:r + 1 - h], U_SEL)
        ts = P.index[r0:r1]
        np.savez_compressed(cache, Q=Q, ts=(ts - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1s'))
        return cls(Q, 'Rolling empirical', ts)

    def q(self, a):
        j = int(np.clip(np.searchsorted(U_SEL, a), 1, len(U_SEL) - 1))
        w = float(np.clip((a - U_SEL[j - 1]) / (U_SEL[j] - U_SEL[j - 1]), 0, 1))
        return self.Q[:, :, j - 1] * (1 - w) + self.Q[:, :, j] * w

    def interval(self, m, alpha=C.RANGE_ALPHA, H=C.RANGE_HORIZON, lag=1):
        """[q_{alpha/2}, q_{1-alpha/2}] (bp) used at decision k: origin stamped k-lag (one-hour data lag), union
        over the horizons covering hours k+1..k+H."""
        hs = np.arange(lag, min(lag + H, 24))
        lo = pd.Series(self.q(alpha / 2)[:, hs].min(1), index=self.origin_ts)
        hi = pd.Series(self.q(1 - alpha / 2)[:, hs].max(1), index=self.origin_ts)
        st = m.idx - pd.Timedelta(hours=lag)
        return lo.reindex(st).ffill().bfill().values, hi.reindex(st).ffill().bfill().values

    def median(self, m, H=C.RANGE_HORIZON, lag=1):
        lo, hi = self.interval(m, 1.0, H, lag)       # alpha = 1: min / max of the medians
        return (lo + hi) / 2


def load_forecasters(m):
    """Neural and statistical forecasters from QDIR, plus the rolling empirical benchmark over m's window."""
    fc = {}
    for name in C.FORECAST_ARTIFACTS:
        p = C.QDIR / f"q_{name.lower().replace(' ', '_')}.npz"
        if p.exists():
            fc[name] = QForecast.from_npz(p, name)
        else:
            print(f'forecasts: missing {p} (run extract_quantiles.py)')
    fc['Rolling empirical'] = QForecast.rolling_empirical(m.idx[0] - pd.Timedelta(hours=2), m.idx[-1] + pd.Timedelta(hours=1))
    return fc
