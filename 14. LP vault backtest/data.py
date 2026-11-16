"""Data loading and signal construction for the LP vault backtest.

All paths are relative to the repository root (REPO env var, default: parent of this folder).
Parquet files are read with pandas/pyarrow when available, else with the bundled pure-Python
reader (pq.py), so the backtest runs without pyarrow.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from lpsim import Market
from strategies import episodes_from_flags, depeg_events

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('REPO', HERE.parent))
QDIR = Path(os.environ.get('QDIR', HERE / 'forecast_quantiles'))
OUT = Path(os.environ.get('OUT', HERE / 'results'))

KUMO_DIR = REPO / '8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_lead6_24_10_15bps_2026-10-19/cv_results_report'
POOL = '0x3416cf6c708da44db2624d63ea0aaef7113527c6'  # USDC/USDT 0.01% Uniswap V3
CUTOFF = 0.09446171671152116  # NP cutoff of the CV-selected Kumo alpha=1 (eta=0.08), fixed before holdout
START, END = '2025-01-15 16:00', '2026-05-04 06:00'  # common window covered by classifier + forecaster


def read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        from pq import read_parquet as rp
        return rp(str(path))


_panel = None


def panel():
    global _panel
    if _panel is None:
        cache = HERE / '.panel_cache.pkl'
        if cache.exists():
            _panel = pd.read_pickle(cache)
        else:
            m = read_parquet(REPO / '4. Stablecoin liquidity ownership/USDC_USDT_hourly_metrics.parquet')
            p = m[m['pool'] == POOL].copy()
            p['ts'] = pd.to_datetime(p['hour'], unit='s', utc=True)
            p = p.set_index('ts').sort_index()
            _panel = p[['open', 'high', 'low', 'close', 'depeg_bps', 'tick', 'liquidity', 'tvlUSD',
                        'volumeUSD', 'feesUSD', 'swap_count']]
            _panel.to_pickle(cache)
    return _panel


def make_market(start=START, end=END):
    return Market(panel(), start, end)


def kumo_scores():
    s = read_parquet(KUMO_DIR / 'holdout_cache/threshold_15/kumo_tabular_alpha_1/operational_scores.parquet')
    return s.set_index('timestamp')


def kumo_alert(m, lag=1, cutoff=CUTOFF, quiet=12):
    """Alert-episode state (12h quiet reset), usable at decision k = state at stamp k-lag."""
    s = kumo_scores()
    s = s[s['split'] == 'test']['probability']
    P = panel()
    raw = (s.reindex(P.index) > cutoff).fillna(False).values
    act = pd.Series(episodes_from_flags(raw, quiet), index=P.index)
    return act.shift(lag).fillna(False).reindex(m.idx).values.astype(bool)


def price_alert(m, theta, lag=0, quiet=12):
    d = panel()['depeg_bps']
    act = pd.Series(episodes_from_flags((d.abs() >= theta).values, quiet), index=d.index)
    return act.shift(lag).fillna(False).reindex(m.idx).values.astype(bool)


def events(m, thr=15.0):
    P = panel()
    ev = depeg_events(P['depeg_bps'].values, P.index, thr)
    t = P.index[ev]
    return t[(t >= m.idx[0]) & (t <= m.idx[-1])]


def oracle_alert(m, thr=15.0, pre=24):
    a = np.zeros(m.n, bool)
    for t in events(m, thr):
        i = m.idx.get_loc(t); a[max(0, i - pre):i] = True
    return a | episodes_from_flags(np.abs(m.depeg) >= thr, 12)


def event_mask(m, thr=15.0, pre=24, post=48):
    a = np.zeros(m.n, bool)
    for t in events(m, thr):
        i = m.idx.get_loc(t); a[max(0, i - pre):i + post] = True
    return a


def kumo_cv_alert_2024(m, duration=21):
    """2024 CV period: the paper's out-of-fold Kumo alert onsets (11/18 events), with episode
    length fixed at the holdout median (hourly CV scores of the main run are not stored)."""
    on = pd.read_csv(KUMO_DIR / 'cv_alert_outcomes.csv')
    on = on[(on.target_threshold == 15) & (on.alpha == 1.0) & (on.model_name == 'kumo_tabular')]
    P = panel(); a = pd.Series(False, index=P.index)
    for t in pd.to_datetime(on.alert_time, utc=True):
        i = P.index.get_loc(t); a.iloc[i:i + duration] = True
    return a.shift(1).fillna(False).reindex(m.idx).values.astype(bool)
