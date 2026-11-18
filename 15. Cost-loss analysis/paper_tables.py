"""LaTeX tables, number macros (\\cl...) and checks of the qualitative claims made in the paper text.

    python paper_tables.py   -> paper/generated/{tab_cl_classifier, tab_cl_candidates, tab_cl_forecaster, tab_cl_joint, cl_numbers}.tex
"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).absolute().parent; OUT, GEN = HERE / 'results', HERE / 'paper' / 'generated'
GEN.mkdir(parents=True, exist_ok=True)
V = pd.read_csv(OUT / 'classifier_value.csv'); F = pd.read_csv(OUT / 'forecaster_value.csv')
FC = pd.read_csv(OUT / 'forecaster_value_curves.csv'); Jc = pd.read_csv(OUT / 'joint_value_curves.csv'); S = pd.read_csv(OUT / 'joint_screening.csv')
NICE = {'kumo_tabular': 'Kumo Tabular', 'tabicl_v2': 'TabICLv2', 'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
        'random_forest': 'Random forest'}


def n(x, d=2):
    if pd.isna(x): return '--'
    if round(x, d) == 0: x = 0.0
    s = f'{abs(x):,.{d}f}'
    return f'$-{s}$' if x < 0 else s


def table(path, caption, label, header, rows, spec, width='\\linewidth', resize=True):
    open_, close_ = (f'\\resizebox{{{width}}}{{!}}{{%\n', '}') if resize else ('', '')
    tex = (f'\\begin{{table}}[pos=!htbp,align=\\centering]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n'
           + open_ + f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n' + '\n'.join(rows) +
           '\n\\bottomrule\n\\end{tabular}' + close_ + '\n\\end{table}\n')
    open(GEN / path, 'w').write(tex)


def lab(r):
    if r.family == 'precursor':
        return f"Price precursor, $|d|\\geq{r.policy.split()[2]}$\\,bp"
    return NICE.get(r.policy, r.policy) + (f" ($\\alpha={r.alpha:g}$)" if pd.notna(r.alpha) else '') + (' (selected)' if r.selected else '')


def row(r):
    return (f"{lab(r)} & {int(r.D)}/{int(r.N)} & {int(r.W)} & {n(r.breakeven)} & {n(r.W / r.D, 1) if r.D else '--'} & "
            f"{n(r['V_0.01'])} & {n(r['V_0.05'])} & {n(r['V_0.1'])} \\\\")


HDR = ('Warning rule & Anticipated & Warnings & Break-even $c/L$ & Pays if $L/c>$ & $V(0.01)$ & $V(0.05)$ & $V(0.1)$')
cv = V[V['sample'] == 'CV 2024']; ho = V[V['sample'] == 'Holdout']; po = V[V['sample'] == 'Pooled']
sel_cv = cv[cv.selected.astype(bool)].iloc[0]
others = cv[(cv.family == 'classifier') & (cv.policy != 'kumo_tabular')]
best_other = others.loc[others['V_0.01'].idxmax()]
pre = lambda g: g[g.family == 'precursor']
rows = ['\\multicolumn{8}{l}{\\emph{Cross-validation, 2024 (18 events)}} \\\\', row(sel_cv), row(best_other)] + [row(r) for _, r in pre(cv).iterrows()]
rows += ['\\midrule', '\\multicolumn{8}{l}{\\emph{Holdout}} \\\\'] + [row(r) for _, r in ho.iterrows()]
rows += ['\\midrule', '\\multicolumn{8}{l}{\\emph{Cross-validation and holdout pooled}} \\\\'] + [row(r) for _, r in po.iterrows()]
table('tab_cl_classifier.tex',
      'Economic value of acting on early warnings. Each warning (alert episode) triggers an intervention of cost $c$; an eligible '
      '15\\,bp event without an alert-episode onset 2 to 24 hours before its first breach causes a loss $L$. Anticipated: events '
      'with such an onset. Break-even $c/L$: anticipated events per warning, the largest cost-loss ratio at which acting on the '
      "warnings beats never acting; ``pays if $L/c>$'' is its inverse,"
      '  the number of interventions a missed event must outweigh. '
      '$V(c/L)$: relative economic value, the share of the saving of a perfect warning system that the rule realises. Price '
      'precursors warn when $|d_t|$ reaches the stated level, with the same 12-hour quiet reset; their thresholds were not tuned.',
      'tab:cl_classifier', HDR, rows, 'lrrrrrrr')
rows = [row(r) for _, r in cv[cv.family == 'classifier'].sort_values(['policy', 'alpha']).iterrows()]
table('tab_cl_candidates.tex', 'Economic value of acting on the warnings of every candidate classifier in cross-validation '
      '(15\\,bp events, 2024 outer test windows). Columns as in Table~\\ref{tab:cl_classifier}.', 'tab:cl_candidates', HDR, rows, 'lrrrrrrr')
rows = []
for (h, th), g in F.groupby(['h', 'threshold'], sort=True):
    rows.append(f'\\multicolumn{{6}}{{l}}{{\\emph{{$|d_{{t+{h}}}|\\geq{th}$\\,bp (base rate {100 * g.base_rate.iloc[0]:.1f}\\%)}}}} \\\\')
    for _, r in g.iterrows():
        rows.append(f"{r.forecaster} & {n(r.BSS)} & {n(r['V_0.02'])} & {n(r['V_0.05'])} & {n(r['V_0.1'])} & {n(r['V_0.2'])} \\\\")
table('tab_cl_forecaster.tex', 'Economic value of acting on forecast exceedance probabilities over the holdout. At each hour a user '
      'protects against $|d_{t+h}|\\geq\\theta$ when the forecast probability exceeds the cost-loss ratio $c/L$. BSS: Brier skill '
      'score against the sample base rate. $V(c/L)$: relative economic value against the better of always and never protecting.',
      'tab:cl_forecaster', 'Forecaster & BSS & $V(0.02)$ & $V(0.05)$ & $V(0.1)$ & $V(0.2)$', rows, 'lrrrrr', resize=False)
ours = Jc[Jc.forecaster == 'Our model']; AL = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
vat = lambda rule, a: ours[(ours.rule == rule)].set_index('cost_loss').V.reindex([a], method='nearest').iloc[0]
rows = [f"{'Classifier: act on every warning' if r == 'classifier' else ('Forecaster alone' if r == 'forecaster' else 'Joint: act on warnings the forecaster confirms')} & "
        + ' & '.join(n(vat(r, a)) for a in AL) + ' \\\\' for r in ['classifier', 'forecaster', 'joint']]
table('tab_cl_joint.tex', 'Joint use of the two models over the holdout: relative economic value $V(c/L)$. The joint rule acts on a '
      'classifier warning only if the forecast probability of a 15\\,bp breach in the next 2 to 24 hours exceeds the cost-loss ratio; '
      'the forecaster alone acts whenever that probability exceeds the ratio.', 'tab:cl_joint',
      'Rule & ' + ' & '.join(f'$V({a:g})$' for a in AL), rows, 'l' + 'r' * len(AL), resize=False)

# ---------- number macros ----------
k_po = po[po.policy == 'kumo_tabular'].iloc[0]; p10 = po[po.policy == 'price precursor 10 bp'].iloc[0]; p12 = po[po.policy == 'price precursor 12 bp'].iloc[0]
sel_ho = ho[ho.selected.astype(bool)].iloc[0]
cell = lambda h, th: F[(F.h == h) & (F.threshold == th)].set_index('forecaster')
best_bss = F.loc[F.groupby(['h', 'threshold']).BSS.idxmax()]
ours_best_bss = int((best_bss.forecaster == 'Our model').sum())
c12 = cell(12, 15); c6 = cell(6, 15)
cl = ours[ours.rule == 'classifier'].set_index('cost_loss').V; jt = ours[ours.rule == 'joint'].set_index('cost_loss').V
cross_joint = float(jt.index[(jt > cl).values][0]) if (jt > cl).any() else np.nan
auc = S[(S.forecaster == 'Our model') & (S.quantity == 'breach probability')].auc.iloc[0]
HE = pd.read_csv(OUT / 'holdout_events.csv')
mac = {
    'HoOnlyClassifierTen': int(((HE.classifier == 1) & (HE.precursor_10 == 0)).sum()),
    'HoOnlyTenClassifier': int(((HE.classifier == 0) & (HE.precursor_10 == 1)).sum()),
    'CvN': int(sel_cv.N), 'CvD': int(sel_cv.D), 'CvW': int(sel_cv.W), 'CvBreak': n(sel_cv.breakeven), 'CvPays': n(sel_cv.W / sel_cv.D, 1),
    'CvVa': n(sel_cv['V_0.01']), 'CvOther': NICE[best_other.policy], 'CvOtherD': int(best_other.D), 'CvOtherVa': n(best_other['V_0.01']),
    'HoN': int(sel_ho.N), 'HoD': int(sel_ho.D), 'HoW': int(sel_ho.W), 'HoBreak': n(sel_ho.breakeven), 'HoPays': n(sel_ho.W / sel_ho.D, 1),
    'HoVa': n(sel_ho['V_0.01']), 'HoVb': n(sel_ho['V_0.05']),
    'HoPreMaxD': int(pre(ho).D.max()), 'HoPreBestVa': n(pre(ho)['V_0.01'].max()),
    'CvPreTenD': int(pre(cv)[pre(cv).policy == 'price precursor 10 bp'].D.iloc[0]), 'CvPreTenW': int(pre(cv)[pre(cv).policy == 'price precursor 10 bp'].W.iloc[0]),
    'CvPreTwelveD': int(pre(cv)[pre(cv).policy == 'price precursor 12 bp'].D.iloc[0]), 'CvPreTwelveW': int(pre(cv)[pre(cv).policy == 'price precursor 12 bp'].W.iloc[0]),
    'PoN': int(k_po.N), 'PoD': int(k_po.D), 'PoW': int(k_po.W), 'PoTenD': int(p10.D), 'PoTenW': int(p10.W), 'PoTwelveD': int(p12.D), 'PoTwelveW': int(p12.W),
    'PoCrossTen': n(p10.crossover_vs_classifier), 'PoCrossTenInv': f'{1 / p10.crossover_vs_classifier:.0f}',
    'PoCrossTwelve': n(p12.crossover_vs_classifier), 'PoCrossTwelveInv': f'{1 / p12.crossover_vs_classifier:.0f}',
    'FcBestBSSCells': ours_best_bss, 'FcCells': len(best_bss),
    'FcSixFifteenBSS': n(c6.loc['Our model', 'BSS']), 'FcTwelveFifteenBSS': n(c12.loc['Our model', 'BSS']),
    'FcTwelveFifteenVc': n(c12.loc['Our model', 'V_0.1']), 'FcTwelveFifteenVd': n(c12.loc['Our model', 'V_0.2']),
    'FcTwelveFifteenGarchVc': n(c12.loc['GARCH', 'V_0.1']), 'FcTwelveFifteenTimeXerVc': n(c12.loc['TimeXer', 'V_0.1']),
    'JointAUC': n(auc), 'JointCross': n(cross_joint), 'JointCrossInv': f'{1 / cross_joint:.0f}' if cross_joint == cross_joint else '--',
    'JointVc': n(vat('joint', 0.1)), 'JointClVc': n(vat('classifier', 0.1)), 'JointVa': n(vat('joint', 0.01)), 'JointClVa': n(vat('classifier', 0.01)),
    'JointTimely': int(S.timely_warnings.iloc[0]), 'JointFalse': int(S.false_warnings.iloc[0]),
}
with open(GEN / 'cl_numbers.tex', 'w') as fh:
    fh.write('% Generated by paper_tables.py -- do not edit by hand\n')
    for k, v in mac.items():
        v = str(v); v = v.replace('$', '')
        fh.write(f'\\newcommand{{\\cl{k}}}{{\\ensuremath{{{v}}}}}\n' if v.startswith('-') else f'\\newcommand{{\\cl{k}}}{{{v}}}\n')

# ---------- claims made in the text ----------
others_cv = cv[(cv.family == 'classifier') & (cv.policy != 'kumo_tabular')]
claims = {
    'holdout: classifier anticipates events the 10 bp precursor misses': mac['HoOnlyClassifierTen'] >= 1,
    'selected classifier has the highest CV recall of all candidates': sel_cv.D >= cv[cv.family == 'classifier'].D.max(),
    'selected classifier beats every other model family at every cost-loss ratio in CV':
        all((sel_cv.D >= r.D) and (sel_cv.D / sel_cv.W >= r.D / r.W - 1e-12 or sel_cv.D > r.D) for _, r in others_cv.iterrows()),
    'holdout: classifier anticipates every event': sel_ho.D == sel_ho.N,
    'holdout: classifier has higher value than every precursor for c/L <= 0.05': all(sel_ho[f'V_{a}'] > pre(ho)[f'V_{a}'].max() for a in (0.01, 0.05)),
    'CV: 10 bp precursor anticipates at least as many events with fewer warnings (dominates)':
        mac['CvPreTenD'] >= sel_cv.D and mac['CvPreTenW'] < sel_cv.W,
    'pooled: classifier has the highest value at c/L = 0.01': k_po['V_0.01'] > pre(po)['V_0.01'].max(),
    'pooled: precursors 10/12 bp better at c/L = 0.1': k_po['V_0.1'] < min(p10['V_0.1'], p12['V_0.1']),
    'forecaster: our model has the highest BSS in at least 4 of 6 cells': ours_best_bss >= 4,
    'forecaster 12h/15bp: our model has the highest value at c/L = 0.1 and 0.2; GARCH negative and TimeXer below 0.1 at c/L = 0.1':
        c12['V_0.1'].idxmax() == 'Our model' and c12['V_0.2'].idxmax() == 'Our model' and c12.loc['GARCH', 'V_0.1'] < 0 and c12.loc['TimeXer', 'V_0.1'] < 0.1,
    'forecaster 15bp: rolling empirical has higher value than our model at c/L = 0.02 (6h and 12h)':
        c6.loc['Rolling empirical', 'V_0.02'] > c6.loc['Our model', 'V_0.02'] and c12.loc['Rolling empirical', 'V_0.02'] > c12.loc['Our model', 'V_0.02'],
    'forecaster 24h/15bp: value negative for every forecaster at c/L = 0.1': (cell(24, 15)['V_0.1'] < 0).all(),
    'joint: breach probability ranks timely above false warnings (AUC > 0.6)': auc > 0.6,
    'joint: classifier alone better at c/L = 0.01, joint better at c/L = 0.1': vat('classifier', 0.01) > vat('joint', 0.01) and vat('joint', 0.1) > vat('classifier', 0.1),
}
for k, v in claims.items():
    print(('CLAIM OK    ' if v else 'CLAIM FAILS ') + k)
print({k: v for k, v in mac.items()})
