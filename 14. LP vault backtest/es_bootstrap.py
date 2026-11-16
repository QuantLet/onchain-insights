import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, OUT
from lpsim import Sim, metrics
from strategies import Recenter, Gated
from run_all import load_forecasters, width_strategy, stationary_bootstrap_diff
from es_strategies import ESBand, es_interval, es_tail_distance
m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m); FC = load_forecasters(); f = FC['Our model']
g_es = ka & (es_tail_distance(f, m) > 10); share = g_es.mean(); absd = np.abs(m.depeg)
g_px = ka & (absd >= np.quantile(absd[ka], 1 - share / ka.mean()))
lo5, hi5, med6 = es_interval(f, m, 0.05, 6)
S = {'W': width_strategy(f, m), 'W+K': Gated(width_strategy(f, m), ka, 'freeze'),
     'W+K&ES': Gated(width_strategy(f, m), g_es, 'freeze'), 'W+K&px': Gated(width_strategy(f, m), g_px, 'freeze'),
     'ESB': ESBand(lo5, hi5, med6, 0.5), 'ESB+K&ES': Gated(ESBand(lo5, hi5, med6, 0.5), g_es, 'freeze'),
     'R': Recenter(2), 'R+K': Gated(Recenter(2), ka, 'freeze'), 'R+K&ES': Gated(Recenter(2), g_es, 'freeze'),
     'R+K&px': Gated(Recenter(2), g_px, 'freeze')}
D = {}; rows = []
for k, s in S.items():
    df = sim.run(s); D[k] = df['hedged'].resample('1D').sum() / 100.0   # bp of $1M per day
    r = metrics(df, m, label=k, event_mask=em); rows.append(r)
print(pd.DataFrame(rows)[['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps']].round(2).to_string())
out = []
for a, b in [('W+K&ES', 'W+K'), ('W+K&ES', 'W'), ('W+K&ES', 'W+K&px'), ('ESB', 'W'), ('ESB+K&ES', 'W+K'),
             ('R+K&ES', 'R+K'), ('R+K&ES', 'R'), ('R+K&ES', 'R+K&px')]:
    mu, lo, hi, p = stationary_bootstrap_diff(D[a], D[b], B=2000)
    out.append(dict(a=a, b=b, diff_bp_yr=mu * 365, ci_lo=lo * 365, ci_hi=hi * 365, p=p))
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv(OUT / 'table11c_es_bootstrap.csv')
