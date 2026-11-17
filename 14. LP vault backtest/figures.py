"""Paper figures from results/ (PNG, transparent background, no grids, legends outside the plot area)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import config as C

INK, INK2 = '#0b0b0b', '#52514e'
EVENT_C = '#e34948'
STYLE = {  # vault -> (label, color, linestyle, linewidth); identical in every figure
    'Gamma': ('Gamma-style ±2 (no model)', '#8a8984', '-', 1.5),
    'Gamma + classifier': ('Gamma-style + classifier freeze', '#2a78d6', '-', 1.8),
    'Gamma + price trigger': ('Gamma-style + 15 bp price trigger', '#eb6834', (0, (4, 2)), 1.6),
    'Forecaster': ('Forecaster range', '#1baf7a', (0, (6, 2, 1.5, 2)), 1.8),
    'Forecaster + classifier': ('Forecaster range + classifier freeze', '#6250d6', '-', 2.2),
    'Forecaster + price trigger': ('Forecaster range + 15 bp price trigger', '#eb6834', (0, (4, 2)), 1.6),
}
ALERT_C, TRIG_C = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False,
                     'savefig.transparent': True, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})
BP = 1e4 / C.V0


def save(fig, name):
    C.FIG.mkdir(exist_ok=True)
    fig.savefig(C.FIG / name, bbox_inches='tight', transparent=True); plt.close(fig); print('saved', C.FIG / name)


def shade(ax, a, idx, color, alpha):
    a = np.asarray(a, int); on = np.flatnonzero(np.diff(np.r_[0, a, 0]) == 1); off = np.flatnonzero(np.diff(np.r_[0, a, 0]) == -1)
    for s, e in zip(on, off):
        ax.axvspan(idx[s], idx[min(e, len(idx) - 1)], color=color, alpha=alpha, lw=0)


def line_handle(k):
    lab, col, ls, lw = STYLE[k]; return Line2D([], [], color=col, ls=ls, lw=lw, label=lab)


def legend_below(fig, handles, ncol=3, y=-0.02):
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=7.5, handlelength=3)


def load_curves(path):
    """Inverse of backtest.save_curves: (DatetimeIndex, {key: DataFrame}, extras)."""
    z = np.load(path, allow_pickle=False)
    idx = pd.to_datetime(z['ts'], unit='s', utc=True); cols = list(z['cols'])
    curves = {k: pd.DataFrame({c: z[f'{k}::{c}'] for c in cols}, index=idx) for k in z['keys']}
    extra = {k: z[k] for k in z.files if '::' not in k and k not in ('ts', 'keys', 'cols')}
    return idx, curves, extra


idx, cur, X = load_curves(C.OUT / 'curves.npz'); dep, alert, trig, ev = X['depeg'], X['alert'], X['price_trigger'], X['events']
PROG = ['Gamma', 'Forecaster', 'Forecaster + classifier']

# ---- Figure 1: depeg path and alerts; drawdowns of the three steps of the framework ------------------------
fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5], hspace=0.08))
a0.plot(idx, dep, color=INK2, lw=0.6)
for y in (C.EVENT_BP, -C.EVENT_BP): a0.axhline(y, color=INK2, lw=0.6, ls=(0, (2, 2)))
shade(a0, alert, idx, ALERT_C, 0.18)
for i in ev: a0.axvline(idx[i], color=EVENT_C, lw=0.8)
a0.set_ylabel('Depeg (bp)')
for k in PROG:
    c = cur[k]['hedged'].cumsum() * BP; dd = (c - c.cummax()).resample('1D').min(); lab, col, ls, lw = STYLE[k]
    a1.plot(dd.index, dd.values, color=col, ls=ls, lw=lw)
for i in ev: a1.axvline(idx[i], color=EVENT_C, lw=0.8, alpha=0.5)
a1.set_ylabel('Drawdown of hedged PnL\n(bp, daily minimum)'); a1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
legend_below(fig, [line_handle(k) for k in PROG] + [Patch(color=ALERT_C, alpha=0.25, label='classifier alert episode'),
             Line2D([], [], color=EVENT_C, lw=0.8, label='15 bp event'), Line2D([], [], color=INK2, lw=0.6, ls=(0, (2, 2)), label='±15 bp')])
save(fig, 'lp_drawdowns.png')

# ---- Figure 2: fees, LVR and costs inside and outside the event windows ------------------------------------
G = pd.read_csv(C.OUT / 'regimes.csv')
G['where'] = np.where(G.regime == 'event windows', 'event', 'outside')
A = G.groupby(['vault', 'where'])[['fees', 'lvr', 'costs', 'net']].sum()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.25))
comps = [('fees', 'Fees'), ('lvr', 'LVR'), ('costs', 'Costs'), ('net', 'Net')]
for ax, (wh, title) in zip(axs, [('event', f'Within the {len(ev)} event windows'), ('outside', 'Outside the event windows')]):
    x = np.arange(len(comps)); bw = 0.27
    vals_all = [[A.loc[(k, wh), c] for c, _ in comps] for k in PROG]
    span = max(map(max, vals_all)) - min(map(min, vals_all)); off = 0.02 * span
    for j, k in enumerate(PROG):
        vals = vals_all[j]; lab, col, ls, lw = STYLE[k]
        ax.bar(x + (j - 1) * bw, vals, bw * 0.92, color=col, edgecolor='none')
        for xi, v in zip(x + (j - 1) * bw, vals):
            ax.text(xi, v + (off if v >= 0 else -off), f'{v:.0f}'.replace('-', '\u2212'), ha='center', rotation=90,
                    va='bottom' if v >= 0 else 'top', fontsize=6, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(x, [l for _, l in comps]); ax.set_title(title, fontsize=8.5, loc='left', color=INK)
    ax.margins(y=0.22)
axs[0].set_ylabel('bp of vault capital (holdout total)')
legend_below(fig, [Patch(color=STYLE[k][1], label=STYLE[k][0]) for k in PROG], y=-0.05)
save(fig, 'lp_fees_lvr.png')

# ---- Figure 3: two events: cumulative fees and LVR ---------------------------------------------------------
sel = [i for i in ev if str(idx[i]).startswith('2025-02-03') or str(idx[i]).startswith('2026-02-04')] or list(ev[:2])
fig, ax = plt.subplots(3, len(sel), figsize=(7.2, 5.6), sharex='col', gridspec_kw=dict(height_ratios=[1, 1.2, 1.2], hspace=0.12, wspace=0.25))
ax = np.atleast_2d(ax).reshape(3, len(sel))
for j, i in enumerate(sel):
    sl = slice(max(0, i - 36), i + 60); ix = idx[sl]
    a0, a1, a2 = ax[0, j], ax[1, j], ax[2, j]
    a0.plot(ix, dep[sl], color=INK2, lw=0.9)
    for y in (C.EVENT_BP, -C.EVENT_BP): a0.axhline(y, color=INK2, lw=0.6, ls=(0, (2, 2)))
    shade(a0, alert[sl], ix, ALERT_C, 0.18)
    a0.set_title(f'Event of {idx[i]:%d %b %Y}', fontsize=8.5, loc='left', color=INK)
    for k in PROG:
        lab, col, ls, lw = STYLE[k]
        a1.plot(ix, cur[k]['fees'].iloc[sl].cumsum().values * BP, color=col, ls=ls, lw=lw)
        a2.plot(ix, cur[k]['lvr'].iloc[sl].cumsum().values * BP, color=col, ls=ls, lw=lw)
    for a in (a0, a1, a2): a.axvline(idx[i], color=EVENT_C, lw=0.9)
    a2.xaxis.set_major_locator(mdates.DayLocator()); a2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    if j == 0:
        a0.set_ylabel('Depeg (bp)'); a1.set_ylabel('Cumulative fees (bp)'); a2.set_ylabel('Cumulative LVR (bp)')
legend_below(fig, [line_handle(k) for k in PROG] + [Patch(color=ALERT_C, alpha=0.25, label='classifier alert episode'),
             Line2D([], [], color=EVENT_C, lw=0.9, label='first 15 bp breach')], y=0.0)
save(fig, 'lp_event_zoom.png')

# ---- Figure 4: March 2023 stress scenario (losses only) ----------------------------------------------------
sidx, sc, SX = load_curves(C.OUT / 'stress_curves.npz'); sdep, first = SX['depeg'], int(SX['first'])
sc = {tuple(k.split('|')): v for k, v in sc.items()}
w0, w1 = pd.Timestamp(C.STRESS_SUM[0], tz='UTC'), pd.Timestamp('2023-03-16', tz='UTC')
fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5], hspace=0.08))
d = pd.Series(sdep, index=sidx).loc[w0:w1]
a0.plot(d.index, d.values, color=INK2, lw=0.9); a0.axvline(sidx[first], color=EVENT_C, lw=0.9); a0.set_ylabel('Depeg (bp)')
lead = max(C.STRESS_LEADS)
SS = [(('Gamma', 'no freeze'), 'Gamma-style ±2, no freeze', STYLE['Gamma']),
      (('Forecaster (rolling empirical)', 'no freeze'), 'Forecaster range (rolling empirical), no freeze', STYLE['Forecaster']),
      (('Gamma', 'at breach'), 'Gamma-style, freeze at the 15 bp breach', STYLE['Gamma + price trigger']),
      (('Gamma', f'{lead}h before'), f'Gamma-style, freeze {lead} h before the breach', STYLE['Gamma + classifier'])]
hs = []
for key, lab, (_, col, ls, lw) in SS:
    c = sc[key]; loss = (-(c['lvr'] - c['gas'] - c['swapcost']).cumsum() * BP).loc[w0:w1]; loss = loss - loss.iloc[0]
    a1.plot(loss.index, np.maximum(loss.values, 0.05), color=col, ls=ls, lw=lw); hs.append(Line2D([], [], color=col, ls=ls, lw=lw, label=lab))
a1.axvline(sidx[first], color=EVENT_C, lw=0.9)
a1.set_yscale('log'); a1.set_ylim(0.05, 8e3); a1.set_ylabel('Cumulative loss, LVR + costs\n(bp, log scale)')
a1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
legend_below(fig, hs + [Line2D([], [], color=EVENT_C, lw=0.9, label='first 15 bp breach')], ncol=2)
save(fig, 'lp_stress_march2023.png')

# ---- Figure 5 (appendix): placebo distribution of the event drawdown ---------------------------------------
P = pd.read_csv(C.OUT / 'placebo.csv'); M = pd.read_csv(C.OUT / 'main.csv', index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, base in zip(axs, ['Gamma', 'Forecaster']):
    ax.hist(P[P.vault == base].event_dd, bins=30, color='#c3c2b7', edgecolor='none')
    for k in [base, f'{base} + price trigger', f'{base} + classifier']:
        lab, col, ls, lw = STYLE[k]; ax.axvline(M.loc[k, 'event_dd'], color=col, ls=ls, lw=lw + 0.4)
    ax.set_title('Gamma-style ±2' if base == 'Gamma' else 'Forecaster range', fontsize=8.5, loc='left', color=INK)
    ax.set_xlabel(f'Event drawdown, sum over {len(ev)} events (bp)')
axs[0].set_ylabel('Placebo schedules')
legend_below(fig, [Patch(color='#c3c2b7', label='classifier episodes at random dates'),
                   Line2D([], [], color=INK2, lw=1.8, label='no freeze (gray: Gamma, green: forecaster)'),
                   Line2D([], [], color='#eb6834', ls=(0, (4, 2)), lw=2, label='15 bp price trigger'),
                   Line2D([], [], color='#2a78d6', lw=2.2, label='classifier (blue: Gamma, violet: forecaster)')], ncol=2, y=-0.12)
save(fig, 'lp_placebo.png')
