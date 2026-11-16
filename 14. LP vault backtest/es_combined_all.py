import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, OUT
from lpsim import Sim, metrics
from strategies import Gated
from run_all import load_forecasters, stationary_bootstrap_diff
from es_strategies import ESBand, es_interval, es_tail_distance
m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m); FC = load_forecasters()
rows, D = [], {}
for nm in ['Our model', 'TimeXer', 'TiDE', 'GARCH', 'Rolling empirical']:
    f = FC[nm]; lo5, hi5, med6 = es_interval(f, m, 0.05, 6); g = ka & (es_tail_distance(f, m) > 10)
    df = sim.run(Gated(ESBand(lo5, hi5, med6, 0.5), g, 'freeze'))
    r = metrics(df, m, label=nm, event_mask=em); r['alert_share'] = g.mean(); rows.append(r)
    D[nm] = df['hedged'].resample('1D').sum() / 100
R = pd.DataFrame(rows); R.to_csv(OUT / 'table12_es_combined_all.csv')
print(R[['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps', 'hedged_cvar5_daily_bps', 'rebal_per_month', 'alert_share']].round(2).to_string())
for o in ['TimeXer', 'TiDE', 'GARCH', 'Rolling empirical']:
    mu, lo, hi, p = stationary_bootstrap_diff(D['Our model'], D[o], B=2000)
    print(f'Our model vs {o}: {mu*365:+.1f} bp/yr [{lo*365:+.1f}, {hi*365:+.1f}] p={p:.3f}')
