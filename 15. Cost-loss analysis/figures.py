"""Figures for the cost-loss analysis (PNG, transparent, no grids, legends below the plots)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).absolute().parent; OUT, FIG = HERE / 'results', HERE / 'figures'
INK, INK2 = '#0b0b0b', '#52514e'
BLUE, ORANGE, AQUA, VIOLET, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#8a8984'
DASH = {8: (0, (1, 1.5)), 10: (0, (4, 2)), 12: (0, (6, 2, 1.5, 2))}
FC_STYLE = {'Our model': (BLUE, '-', 2.2), 'TiDE': (AQUA, (0, (6, 2, 1.5, 2)), 1.6), 'TimeXer': (ORANGE, (0, (4, 2)), 1.6),
            'GARCH': (VIOLET, (0, (1, 1.5)), 1.6), 'Rolling empirical': (GRAY, '-', 1.4)}
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False,
                     'savefig.transparent': True, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})
XLAB = 'Cost-loss ratio $c/L$ (log scale)'


def save(fig, name):
    FIG.mkdir(exist_ok=True); fig.savefig(FIG / name, bbox_inches='tight', transparent=True); plt.close(fig); print('saved', FIG / name)


def axis(ax, ylo=-0.4):
    ax.set_xscale('log'); ax.axhline(0, color=INK2, lw=0.6); ax.set_ylim(ylo, 1.0); ax.set_xlim(0.002, 0.5)
    ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3], ['0.003', '0.01', '0.03', '0.1', '0.3'])


def legend_below(fig, handles, ncol=3, y=-0.02):
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=7.5, handlelength=3)


# ---- Figure 1: value of acting on classifier warnings ------------------------------------------------------
V = pd.read_csv(OUT / 'classifier_value.csv'); Cc = pd.read_csv(OUT / 'classifier_value_curves.csv')
cv = V[(V['sample'] == 'CV 2024') & (V.family == 'classifier') & (V.policy != 'kumo_tabular')]
alt = cv.loc[cv['V_0.01'].idxmax()]                          # best classifier from another model family in CV
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True, gridspec_kw=dict(wspace=0.18))
for ax, s in zip(axs, ['CV 2024', 'Holdout', 'Pooled']):
    g = Cc[Cc['sample'] == s]
    k = g[g.selected.astype(bool) & (g.family == 'classifier')]
    ax.plot(k.cost_loss, k.V, color=BLUE, lw=2.2)
    if s == 'CV 2024':
        a = g[(g.policy == alt.policy) & (g.model_alpha == alt.alpha)]; ax.plot(a.cost_loss, a.V, color=VIOLET, lw=1.6)
    for th, ls in DASH.items():
        p = g[g.policy == f'price precursor {th} bp']; ax.plot(p.cost_loss, p.V, color=ORANGE, ls=ls, lw=1.6)
    axis(ax); ax.set_title(s if s != 'Pooled' else 'CV 2024 and holdout pooled', fontsize=8.5, loc='left', color=INK); ax.set_xlabel(XLAB)
axs[0].set_ylabel('Relative economic value')
nice = {'tabicl_v2': 'TabICLv2', 'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'random_forest': 'Random forest'}
legend_below(fig, [Line2D([], [], color=BLUE, lw=2.2, label='Early-warning classifier (selected)'),
                   Line2D([], [], color=VIOLET, lw=1.6, label=f'Best other model family in CV ({nice.get(alt.policy, alt.policy)}, $\\alpha$={alt.alpha:g})')]
             + [Line2D([], [], color=ORANGE, ls=ls, lw=1.6, label=f'Price precursor, |d| $\\geq$ {th} bp') for th, ls in DASH.items()], ncol=3, y=-0.04)
save(fig, 'cl_classifier_value.png')

# ---- Figure 2: value of acting on the forecaster's exceedance probabilities --------------------------------
Fc = pd.read_csv(OUT / 'forecaster_value_curves.csv')
cells = [(6, 10), (12, 10), (6, 15), (12, 15)]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.25))
for ax, (h, th) in zip(axs.flat, cells):
    g = Fc[(Fc.h == h) & (Fc.threshold == th)]
    for nm, (col, ls, lw) in FC_STYLE.items():
        x = g[g.forecaster == nm]; ax.plot(x.cost_loss, x.V, color=col, ls=ls, lw=lw)
    axis(ax, -0.6); ax.set_title(f'$|d_{{t+{h}}}| \\geq {th}$ bp', fontsize=8.5, loc='left', color=INK)
for ax in axs[1]: ax.set_xlabel(XLAB)
for ax in axs[:, 0]: ax.set_ylabel('Relative economic value')
legend_below(fig, [Line2D([], [], color=c, ls=ls, lw=lw, label=nm) for nm, (c, ls, lw) in FC_STYLE.items()], ncol=5, y=0.0)
save(fig, 'cl_forecaster_value.png')

# ---- Figure 3: joint use: classifier warnings screened by the forecaster -----------------------------------
J = pd.read_csv(OUT / 'joint_value_curves.csv'); J = J[J.forecaster == 'Our model']
Sx = pd.read_csv(OUT / 'joint_warnings.csv')
fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.3))
for rule, col, ls, lw in [('classifier', BLUE, '-', 2.0), ('forecaster', AQUA, (0, (6, 2, 1.5, 2)), 1.6), ('joint', VIOLET, '-', 2.4)]:
    x = J[J.rule == rule]; a0.plot(x.cost_loss, x.V, color=col, ls=ls, lw=lw)
p = Cc[(Cc['sample'] == 'Holdout') & (Cc.policy == 'price precursor 10 bp')]; a0.plot(p.cost_loss, p.V, color=ORANGE, ls=DASH[10], lw=1.4)
axis(a0, -0.6); a0.set_xlabel(XLAB); a0.set_ylabel('Relative economic value'); a0.set_title('Holdout', fontsize=8.5, loc='left', color=INK)
rng = np.random.default_rng(0)
for lab, col, mk, y in [('false', GRAY, 'o', 0), ('timely', BLUE, 'D', 1)]:
    z = Sx[Sx.status == lab]; a1.scatter(np.maximum(z.breach_probability, 1e-3), y + rng.uniform(-0.15, 0.15, len(z)), s=16, color=col, marker=mk, lw=0)
a1.set_xscale('log'); a1.set_yticks([0, 1], ['false\nwarnings', 'timely\nwarnings']); a1.set_ylim(-0.6, 1.6)
a1.set_xlabel('Forecast breach probability at the warning'); a1.set_title('Screening the warnings', fontsize=8.5, loc='left', color=INK)
legend_below(fig, [Line2D([], [], color=BLUE, lw=2.0, label='Classifier: act on every warning'),
                   Line2D([], [], color=AQUA, ls=(0, (6, 2, 1.5, 2)), lw=1.6, label='Forecaster alone: act when breach probability > $c/L$'),
                   Line2D([], [], color=VIOLET, lw=2.4, label='Joint: act on warnings with breach probability > $c/L$'),
                   Line2D([], [], color=ORANGE, ls=DASH[10], lw=1.4, label='Price precursor, |d| $\\geq$ 10 bp')], ncol=2, y=-0.04)
save(fig, 'cl_joint_value.png')
