import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, price_alert, OUT
from lpsim import Sim, metrics
from strategies import Recenter, Gated, Combined, episodes_from_flags, episode_onsets
from run_all import load_forecasters, width_strategy
from es_strategies import ESBand, ESSized, es_interval, es_phi, es_tail_distance
m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m); FC = load_forecasters()
cols = ['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps',
        'hedged_cvar5_daily_bps', 'rebal_per_month', 'event_hedged_bps', 'calm_hedged_apr']
rows = []
def add(s, fc, fam, extra=None):
    r = metrics(sim.run(s), m, label=s.name, event_mask=em); r['forecaster'] = fc; r['family'] = fam
    for kk, vv in (extra or {}).items(): r[kk] = vv
    rows.append(r)
add(Recenter(2), '-', 'benchmark'); add(Gated(Recenter(2), ka, 'freeze', name='Recenter ±2 + Kumo freeze'), '-', 'benchmark')
for nm in ['Our model', 'TimeXer', 'TiDE', 'GARCH', 'Rolling empirical']:
    f = FC[nm]
    W = lambda: width_strategy(f, m, name=f'{nm} width')
    add(W(), nm, 'width (q10-q90, 6h)')
    add(Gated(W(), ka, 'freeze', name=f'{nm} width + Kumo freeze'), nm, 'width + Kumo freeze')
    lo5, hi5, med6 = es_interval(f, m, 0.05, 6)
    for kap in [0.5, 1.0]:
        add(ESBand(lo5, hi5, med6, kap, name=f'{nm} ES band k={kap}'), nm, f'ES5% band x{kap} (6h)')
    add(Gated(ESBand(lo5, hi5, med6, 0.5), ka, 'freeze', name=f'{nm} ES band k=0.5 + Kumo freeze'), nm, 'ES band x0.5 + Kumo freeze')
    phi = es_phi(f, m)
    add(ESSized(W(), phi, name=f'{nm} width, ES-sized'), nm, 'width, ES-sized capital', {'mean_phi': phi.mean()})
    dist = es_tail_distance(f, m)
    for tau in [6, 10]:
        g = ka & (dist > tau)
        g = episodes_from_flags(g, 1)  # keep hourly intersection (no extra quiet period)
        add(Gated(W(), g, 'freeze', name=f'{nm} width + Kumo∧ES>{tau} freeze'), nm, f'width + Kumo AND ES>{tau}bp freeze',
            {'alert_share': g.mean()})
    l1, h1, _ = es_interval(f, m, 0.01, 24)
    add(Combined(W(), ka, l1, h1, name=f'{nm} width + Kumo ES-dislocation range'), nm, 'width + Kumo ES1% dislocation range')
R = pd.DataFrame(rows); R.to_csv(OUT / 'table10_es_strategies.csv')
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200); pd.set_option('display.precision', 1)
print(R[['family'] + cols + ['alert_share', 'mean_phi']].to_string())
