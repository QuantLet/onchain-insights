"""Effect of fee compounding on the main comparison: 'none' (fees held as cash), 'rebalance' (fees reinvested at
each re-mint, Gamma-style), 'threshold' (plus a collect-and-reinvest transaction whenever accrued fees exceed
0.1% of capital)."""
import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, price_alert, OUT
from lpsim import Sim, metrics
from strategies import Static, Recenter, VolRecenter, Gated
from run_all import load_forecasters, width_strategy
m = make_market(); em = event_mask(m); ka = kumo_alert(m); p8 = price_alert(m, 8); FC = load_forecasters()
def specs():
    return [Static(5), Recenter(2), VolRecenter(2),
            Gated(Recenter(2), ka, 'freeze', name='Recenter ±2 + Kumo freeze'),
            Gated(Recenter(2), p8, 'freeze', name='Recenter ±2 + price trigger freeze'),
            width_strategy(FC['Our model'], m, name='Our model width'),
            width_strategy(FC['TimeXer'], m, name='TimeXer width'),
            width_strategy(FC['TiDE'], m, name='TiDE width'),
            width_strategy(FC['GARCH'], m, name='GARCH width'),
            Gated(width_strategy(FC['Our model'], m), ka, 'freeze', name='Our model width + Kumo freeze')]
rows = []
for mode in ['none', 'rebalance', 'threshold']:
    sim = Sim(m, compound=mode)
    for s in specs():
        r = metrics(sim.run(s), m, label=s.name, event_mask=em); r['compounding'] = mode; rows.append(r)
C = pd.DataFrame(rows); C.to_csv(OUT / 'table9_compounding.csv')
pd.set_option('display.width', 250); pd.set_option('display.precision', 1)
piv = C.pivot_table(index=C.index, columns='compounding', values=['hedged_cagr', 'net_vs_rebal_apr', 'hedged_sharpe', 'hedged_maxdd_bps'], sort=False)
print(piv.to_string())
print(C[C.compounding=='threshold'][['compounds_per_month','rebal_per_month']].to_string())
