"""Matched-width comparison: scale each forecaster's q10-q90 (6h) half-width by k, trace net PnL vs average
width and vs risk; constant-width recentering is the no-information frontier."""
import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, OUT
from lpsim import Sim, metrics
from strategies import Recenter, ForecastWidth, Gated
from run_all import load_forecasters
m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m)
FC = load_forecasters()
rows = []
for w in [1, 2, 3, 4, 5, 6, 8]:
    r = metrics(sim.run(Recenter(w)), m, label=f'Recenter ±{w}', event_mask=em); r['forecaster'] = 'Constant width'; r['k'] = w; rows.append(r)
for nm, f in FC.items():
    lo, hi = f.interval(m, 0.2, 6); med = f.median_path(m, 6); mid = (lo + hi) / 2; half = (hi - lo) / 2
    for k in [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]:
        s = ForecastWidth(mid - k * half, mid + k * half, med, name=f'{nm} k={k}')
        r = metrics(sim.run(s), m, label=s.name, event_mask=em); r['forecaster'] = nm; r['k'] = k; rows.append(r)
F = pd.DataFrame(rows); F.to_csv(OUT / 'table6_width_frontier.csv')
# interpolate net PnL, max DD, worst day, CVaR at common average widths
grid = [3, 4, 5, 6, 8]
out = []
for nm, g in F.groupby('forecaster'):
    g = g.sort_values('avg_width_ticks')
    x = g['avg_width_ticks'].values
    rec = {'forecaster': nm}
    for W in grid:
        if x.min() <= W <= x.max():
            for c, lab in [('net_vs_rebal_apr', 'net'), ('hedged_maxdd_bps', 'dd'), ('hedged_sharpe', 'sh')]:
                rec[f'{lab}@{W}'] = np.interp(W, x, g[c].values)
    out.append(rec)
M = pd.DataFrame(out).set_index('forecaster')
pd.set_option('display.width', 250); pd.set_option('display.precision', 1)
print(F[['forecaster', 'k', 'avg_width_ticks', 'net_vs_rebal_apr', 'hedged_sharpe', 'hedged_maxdd_bps', 'worst_day_bps', 'hedged_cvar5_daily_bps', 'rebal_per_month']].to_string())
print('\nInterpolated at common average width (ticks)\n', M[[c for c in M.columns if c.startswith('net')]].round(0).to_string())
print(M[[c for c in M.columns if c.startswith('sh')]].round(2).to_string())
print(M[[c for c in M.columns if c.startswith('dd')]].round(1).to_string())
M.to_csv(OUT / 'table6b_matched_width.csv')

# ---- figure: net PnL vs average width ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                     'grid.color': '#e7e6e2', 'grid.linewidth': 0.6, 'legend.frameon': False})
series = [('Our model', '#2a78d6', 2.2), ('TiDE', '#eb6834', 1.6), ('TimeXer', '#1baf7a', 1.6), ('GARCH', '#eda100', 1.6),
          ('Constant width', '#8a8984', 1.6)]
fig, ax = plt.subplots(figsize=(7.0, 3.8))
for nm, col, lw in series:
    g = F[F.forecaster == nm].sort_values('avg_width_ticks')
    g = g[g.avg_width_ticks <= 14]
    ax.plot(g.avg_width_ticks, g.net_vs_rebal_apr, color=col, lw=lw, marker='o', ms=4,
            ls='--' if nm == 'Constant width' else '-', label=nm if nm != 'Constant width' else 'Constant width (Recenter ±w)')
ax.set_xlabel('Average LP range width (ticks = bp)'); ax.set_ylabel('Net PnL vs rebalancing (bp/yr)')
ax.set_title('Forecast-driven width adds 70-160 bp/yr over constant width below 6 ticks; the forecasters are close',
             fontsize=9, loc='left')
ax.legend(fontsize=7.5, loc='upper right'); ax.set_xlim(2.5, 14)
fig.tight_layout(); fig.savefig(OUT / 'fig4_width_frontier.png', dpi=200); fig.savefig(OUT / 'fig4_width_frontier.pdf')
print('fig4 written')
