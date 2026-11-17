"""Pool panel, markets, alert schedules and depeg events."""
import numpy as np
import pandas as pd
import config as C
from lpsim import Market
from strategies import episodes_from_flags, depeg_events


def read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:                       # no pyarrow: bundled pure-Python reader
        from pq import read_parquet as rp
        return rp(str(path))


_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        cache = C.HERE / '.panel_cache.pkl'
        fresh = cache.exists() and (not C.PANEL.exists() or cache.stat().st_mtime > C.PANEL.stat().st_mtime)
        if fresh:
            _PANEL = pd.read_pickle(cache)
        else:
            m = read_parquet(C.PANEL)
            p = m[m['pool'] == C.POOL].copy()
            p['ts'] = pd.to_datetime(p['hour'], unit='s', utc=True)
            _PANEL = p.set_index('ts').sort_index()[['open', 'high', 'low', 'close', 'depeg_bps', 'liquidity',
                                                      'volumeUSD', 'feesUSD']]
            _PANEL.to_pickle(cache)
    return _PANEL


def market(start, end):
    return Market(panel(), start, end)


def _to_market(flags, m, lag):
    return pd.Series(flags, index=panel().index).shift(lag, fill_value=False).reindex(m.idx, fill_value=False).values.astype(bool)


def kumo_scores():
    s = read_parquet(C.KUMO_SCORES).set_index('timestamp')
    if 'split' in s:
        s = s[s['split'] == 'test']
    s.index = pd.to_datetime(s.index, utc=True)
    return s['probability']


def kumo_alert(m, lag=1):
    """Classifier alert episodes (cutoff + quiet reset); the state stamped k is usable at decision k+1."""
    raw = (kumo_scores().reindex(panel().index) > C.KUMO_CUTOFF).fillna(False).values
    return _to_market(episodes_from_flags(raw, C.KUMO_QUIET), m, lag)


def kumo_cv_alert(m, lag=1):
    """Out-of-fold alert onsets from cross-validation, as episodes of fixed length (hourly CV scores not stored)."""
    on = pd.read_csv(C.KUMO_CV_ALERTS)
    for k, v in C.KUMO_CV_FILTER.items():
        on = on[on[k] == v]
    P = panel(); a = np.zeros(len(P), bool)
    for t in pd.to_datetime(on['alert_time'], utc=True):
        if t in P.index:
            i = P.index.get_loc(t); a[i:i + C.KUMO_CV_EPISODE_H] = True
    return _to_market(a, m, lag)


def price_alert(m, theta=C.EVENT_BP, lag=0):
    """Freeze while |d| >= theta, with the same quiet reset as the classifier (price is observed immediately)."""
    return _to_market(episodes_from_flags((panel()['depeg_bps'].abs() >= theta).values, C.KUMO_QUIET), m, lag)


def events(m, thr=C.EVENT_BP):
    """Positions (in m) of the first breaches of |d| >= thr."""
    ev = depeg_events(panel()['depeg_bps'].values, thr)
    t = panel().index[ev]
    return np.array([m.idx.get_loc(x) for x in t if m.idx[0] <= x <= m.idx[-1]], int)


def oracle_alert(m, pre=24):
    """Perfect foresight: freeze from `pre` hours before each event, and while |d| >= threshold."""
    a = np.zeros(m.n, bool)
    for i in events(m):
        a[max(0, i - pre):i] = True
    return a | episodes_from_flags(np.abs(m.depeg) >= C.EVENT_BP, C.KUMO_QUIET)
