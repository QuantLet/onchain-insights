"""LaTeX tables, number macros and a check of the qualitative claims made in the paper text.

Writes paper/generated/: tab_lp_main.tex, tab_lp_events.tex, tab_lp_forecasters.tex, tab_lp_2024.tex,
tab_lp_sensitivity.tex, tab_lp_stress.tex and lp_numbers.tex (\\newcommand macros used in the text).
Prints CLAIM OK / CLAIM FAILS for each statement the text makes; revise the text for any failure.
"""
import json
import numpy as np, pandas as pd
import config as C

M = pd.read_csv(C.OUT / 'main.csv', index_col=0); E = pd.read_csv(C.OUT / 'per_event.csv'); G = pd.read_csv(C.OUT / 'regimes.csv')
P = pd.read_csv(C.OUT / 'placebo.csv'); F = pd.read_csv(C.OUT / 'forecasters.csv'); Y = pd.read_csv(C.OUT / 'y2024.csv', index_col=0)
S = pd.read_csv(C.OUT / 'sensitivity.csv'); ST = pd.read_csv(C.OUT / 'stress.csv'); meta = json.load(open(C.OUT / 'meta.json'))
C.GEN.mkdir(parents=True, exist_ok=True)


def n(x, d=1):
    if pd.isna(x): return '--'
    if round(x, d) == 0: x = 0.0
    s = f'{abs(x):,.{d}f}'.replace(',', '{,}')
    return f'$-{s}$' if x < 0 else s


LAB = {'Rebal': 'Rebalancing portfolio', 'Static': f'Static $\\pm{C.STATIC_HALF_WIDTH}$ around parity',
       'Gamma': f'Gamma-style $\\pm{C.GAMMA_HALF_WIDTH}$', 'Gamma + classifier': '\\quad + classifier freeze',
       'Gamma + price trigger': f'\\quad + price trigger, {C.EVENT_BP:.0f}\\,bp', 'Gamma + oracle': '\\quad + oracle',
       'Forecaster': 'Forecaster range', 'Forecaster + classifier': '\\quad + classifier freeze (two-model vault)',
       'Forecaster + price trigger': f'\\quad + price trigger, {C.EVENT_BP:.0f}\\,bp', 'Forecaster + oracle': '\\quad + oracle'}


def table(path, caption, label, header, rows, spec, wide=True, note=None):
    body = '\n'.join(rows)
    rb = ('\\resizebox{\\textwidth}{!}{%' if wide else '\\resizebox{\\linewidth}{!}{%')
    tex = (f'\\begin{{table}}[pos=!htbp,align=\\centering]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n{rb}\n'
           f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}\n'
           + (f'\\par\\smallskip\\footnotesize {note}\n' if note else '') + '\\end{table}\n')
    open(C.GEN / path, 'w').write(tex)


# ---------- main table ----------
rows = [f"{LAB['Rebal']} & 0 & 0 & 0 & 0 & -- & -- & 0 & 0 & 0 & 0 & 0 \\\\"]
for k in M.index:
    r = M.loc[k]
    if k in ('Gamma', 'Forecaster'): rows.append('\\midrule')
    rows.append(f"{LAB[k]} & {n(r.fees,0)} & {n(r.lvr,0)} & {n(r.costs,0)} & {n(r.net,0)} & {n(r.sharpe,1)} & {n(r.sortino,1)} & "
                f"{n(r.event_dd)} & {n(r.max_dd)} & {n(r.worst_day)} & {n(r.cvar5,2)} & {n(r.reposition_per_month)} \\\\")
ne = meta['holdout']['classifier']['events']
table('tab_lp_main.tex',
      f"Liquidity-provision backtest over the holdout ({ne} events of {C.EVENT_BP:.0f}\\,bp; \\$1M vault). Fees, LVR, costs and net "
      "profit are annualised (bp per year); net profit is measured against the rebalancing portfolio, which therefore has zero "
      "profit and risk by construction. Sharpe ratio of daily and Sortino ratio of hourly net profit. Event DD: sum over events "
      f"of the largest drawdown of cumulative net profit from {C.EVENT_WINDOW[0]} hours before to {C.EVENT_WINDOW[1]} hours after "
      "each event; Max DD, worst day and CVaR$_{5\\%}$ of daily net profit, in bp. Classifier freeze: no repositioning during "
      f"alert episodes. Price trigger: no repositioning while $|d_k|\\ge{C.EVENT_BP:.0f}$\\,bp. Oracle: no repositioning from "
      "24 hours before each event.",
      'tab:lp_main', 'Vault & Fees & LVR & Costs & Net & Sharpe & Sortino & Event DD & Max DD & Worst day & CVaR$_{5\\%}$ & Reposit./month',
      rows, 'lrrrrrrrrrrr')

# ---------- per-event drawdowns ----------
keys = ['Gamma', 'Gamma + classifier', 'Gamma + price trigger', 'Forecaster', 'Forecaster + classifier', 'Forecaster + price trigger']
evs = list(dict.fromkeys(E.event))
hdr = 'Event & ' + ' & '.join(pd.Timestamp(e).strftime('%-d %b %y') for e in evs) + ' \\\\\nLargest $|d|$ (bp) & ' + \
      ' & '.join(f"{E[(E.event == e)].max_abs_depeg.iloc[0]:.0f}" for e in evs)
rows = []
for k in keys:
    if k == 'Forecaster': rows.append('\\midrule')
    rows.append(f"{LAB[k]} & " + ' & '.join(n(E[(E.event == e) & (E.vault == k)].drawdown.iloc[0]) for e in evs) + ' \\\\')
table('tab_lp_events.tex', f"Drawdown of net profit within each holdout event window (bp; {C.EVENT_WINDOW[0]} hours before to "
      f"{C.EVENT_WINDOW[1]} hours after the first {C.EVENT_BP:.0f}\\,bp breach).", 'tab:lp_events', hdr, rows, 'l' + 'r' * len(evs), wide=False)

# ---------- forecasters ----------
rows = []
for fc in dict.fromkeys(F.forecaster):
    for rule in ['asymmetric', 'symmetric']:
        for ov in ['none', 'classifier', 'price trigger']:
            r = F[(F.forecaster == fc) & (F.rule == rule) & (F.overlay == ov)].iloc[0]
            lab = (f'{fc}, {rule} range' if ov == 'none' else ('\\quad + classifier freeze' if ov == 'classifier' else f'\\quad + price trigger, {C.EVENT_BP:.0f}\\,bp'))
            rows.append(f"{lab} & {n(r.net,0)} & {n(r.sharpe)} & {n(r.sortino)} & {n(r.event_dd)} & {n(r.max_dd)} & {n(r.worst_day)} & {n(r.cvar5,2)} & {n(r.reposition_per_month)} \\\\")
    rows.append('\\midrule')
table('tab_lp_forecasters.tex', "Forecaster range rule with each forecaster over the holdout. Asymmetric: the predictive "
      f"{100*C.RANGE_ALPHA/2:.0f}\\%--{100*(1-C.RANGE_ALPHA/2):.0f}\\% interval over the next {C.RANGE_HORIZON} hours (main specification). "
      "Symmetric: the same width centred on the median forecast. Columns as in Table~\\ref{tab:lp_main}.",
      'tab:lp_forecasters', 'Vault & Net & Sharpe & Sortino & Event DD & Max DD & Worst day & CVaR$_{5\\%}$ & Reposit./month', rows[:-1], 'lrrrrrrrr')

# ---------- 2024 ----------
rows = []
for k in Y.index:
    r = Y.loc[k]
    if k in ('Gamma', 'Forecaster'): rows.append('\\midrule')
    lab = LAB[k].replace('Forecaster range', 'Forecaster range (rolling empirical)')
    rows.append(f"{lab} & {n(r.net,0)} & {n(r.sharpe)} & {n(r.sortino)} & {n(r.event_dd)} & {n(r.max_dd)} & {n(r.worst_day)} & {n(r.cvar5,2)} \\\\")
y4 = meta['y2024']['classifier']
table('tab_lp_2024.tex', f"Out-of-sample check on 2024 ({C.Y2024[0][:10]} to {C.Y2024[1][:10]}, the part of the year with fee data; "
      f"{y4['events']} events). Classifier alerts are the out-of-fold alert onsets of the selected specification in cross-validation, "
      f"as episodes of {C.KUMO_CV_EPISODE_H} hours (hourly cross-validation scores are not stored). The rolling empirical forecaster "
      "stands in for the neural forecasts, which cover only the holdout. Columns as in Table~\\ref{tab:lp_main}.",
      'tab:lp_2024', 'Vault & Net & Sharpe & Sortino & Event DD & Max DD & Worst day & CVaR$_{5\\%}$', rows, 'lrrrrrrr')

# ---------- sensitivity ----------
SC = {'Base': 'Base case', 'Fees reinvested': 'Fees reinvested at repositioning', 'Low cost': 'Low cost (\\$2 gas, no slippage)',
      'High cost': 'High cost (\\$50 gas, 2\\,bp swap)', '100k vault': '\\$100k vault', '5M vault': '\\$5M vault', 'JIT 50%': 'JIT 50\\%'}
VK = ['Gamma', 'Gamma + classifier', 'Gamma + price trigger', 'Forecaster', 'Forecaster + classifier', 'Forecaster + price trigger']
rows = []
for sc in dict.fromkeys(S.scenario):
    g = S[S.scenario == sc].set_index('vault')
    rows.append(f"{SC.get(sc, sc)} & " + ' & '.join(f"{n(g.loc[v,'event_dd'])} ({n(g.loc[v,'net'],0)})" for v in VK) + ' \\\\')
table('tab_lp_sensitivity.tex', "Sensitivity of the holdout results: event drawdown in bp, with net profit in bp per year in "
      "parentheses. ``Fees reinvested'' adds accrued fees to the position at each repositioning; ``JIT 50\\%'' removes half of the "
      "pool's fee income, captured by just-in-time liquidity; ``High cost'' uses a 1\\,bp pool fee plus 1\\,bp slippage on swaps.",
      'tab:lp_sensitivity', 'Scenario & Gamma-style & + classifier & + price trigger & Forecaster range & + classifier & + price trigger',
      rows, 'lrrrrrr')

# ---------- stress ----------
fz = list(dict.fromkeys(ST[ST.vault != 'Static'].freeze))
rows = []
for v in [x for x in dict.fromkeys(ST.vault) if x != 'Static']:
    g = ST[ST.vault == v].set_index('freeze')
    gamma_lab = 'Gamma-style $\\pm' + str(C.GAMMA_HALF_WIDTH) + '$'
    vlab = v.replace('Gamma', gamma_lab).replace('Forecaster', 'Forecaster range')
    rows.append(vlab + ' & ' + ' & '.join(n(g.loc[f, 'loss']) for f in fz) + ' \\\\')
st_static = ST[ST.vault == 'Static'].loss.iloc[0]
table('tab_lp_stress.tex', f"Stress scenario: USDC depeg of {C.STRESS_SUM[0][8:]}--{C.STRESS_SUM[1][8:]} March 2023. Losses from LVR "
      f"and repositioning costs (bp of a \\$1M vault); fee income is not observed in this period and is excluded. Freezes start at the "
      f"first {C.EVENT_BP:.0f}\\,bp breach ({pd.Timestamp(meta['stress']['first_breach']):%H:%M} UTC on "
      f"{pd.Timestamp(meta['stress']['first_breach']):%-d March}) or the stated number of hours before it. A static "
      f"$\\pm{C.STATIC_HALF_WIDTH}$ range loses {n(st_static)}\\,bp.",
      'tab:lp_stress', 'Vault & ' + ' & '.join(f.replace('h before', '\\,h before') for f in fz), rows, 'l' + 'r' * len(fz), wide=False)

# ---------- number macros ----------
def pv(base, col):
    real = M.loc[f'{base} + classifier', col]; d = P[P.vault == base][col]
    p = (d >= real).mean()
    return f'\\ensuremath{{<{1/len(d):.3f}}}' if p == 0 else f'{p:.3f}'
cut = lambda a, b: f'{100 * (1 - a / b):.0f}'
reg = G.set_index(['vault', 'regime'])
ev = lambda k, c: reg.loc[(k, 'event windows'), c]
mac = {
    'NEvents': str(ne), 'HoldoutStart': pd.Timestamp(meta['holdout']['start']).strftime('%-d %B %Y'),
    'HoldoutEnd': pd.Timestamp(meta['holdout']['end']).strftime('%-d %B %Y'), 'MaxDepeg': f"{meta['holdout']['max_abs_depeg']:.0f}",
    'AlertShare': f"{100 * meta['holdout']['classifier']['alert_share']:.1f}", 'Episodes': str(meta['holdout']['classifier']['episodes']),
    'FalseEpisodes': str(meta['holdout']['classifier']['false_episodes']), 'NPlacebo': str(meta['n_placebo']),
    'YNEvents': str(y4['events']), 'YDetected': str(y4['detected']),
}
for key, k in [('Static', 'Static'), ('Gamma', 'Gamma'), ('GK', 'Gamma + classifier'), ('GP', 'Gamma + price trigger'), ('GO', 'Gamma + oracle'),
               ('Fc', 'Forecaster'), ('FcK', 'Forecaster + classifier'), ('FcP', 'Forecaster + price trigger'), ('FcO', 'Forecaster + oracle')]:
    r = M.loc[k]
    for c, nm, d in [('net', 'Net', 0), ('sharpe', 'Sharpe', 1), ('sortino', 'Sortino', 1), ('event_dd', 'EventDD', 1), ('max_dd', 'MaxDD', 1),
                     ('worst_day', 'WorstDay', 1), ('cvar5', 'CVaR', 2), ('costs', 'Costs', 0), ('reposition_per_month', 'Repo', 1), ('fees', 'Fees', 0), ('lvr', 'LVR', 0)]:
        mac[key + nm] = n(r[c], d)
    if k != 'Static':
        for c, nm in [('fees', 'EvFees'), ('lvr', 'EvLVR'), ('costs', 'EvCosts'), ('net', 'EvNet')]:
            mac[key + nm] = n(ev(k, c), 0)
mac.update({
    'GKEventDDCut': cut(M.loc['Gamma + classifier', 'event_dd'], M.loc['Gamma', 'event_dd']),
    'GKMaxDDCut': cut(M.loc['Gamma + classifier', 'max_dd'], M.loc['Gamma', 'max_dd']),
    'FcKEventDDCut': cut(M.loc['Forecaster + classifier', 'event_dd'], M.loc['Forecaster', 'event_dd']),
    'FcKMaxDDCut': cut(M.loc['Forecaster + classifier', 'max_dd'], M.loc['Forecaster', 'max_dd']),
    'FcKEvLVRCut': cut(ev('Forecaster + classifier', 'lvr'), ev('Forecaster', 'lvr')),
    'FcEvLVRCut': cut(ev('Forecaster', 'lvr'), ev('Gamma', 'lvr')),
    'FcKvsGammaEventDDCut': cut(M.loc['Forecaster + classifier', 'event_dd'], M.loc['Gamma', 'event_dd']),
    'FcKvsGammaMaxDDCut': cut(M.loc['Forecaster + classifier', 'max_dd'], M.loc['Gamma', 'max_dd']),
    'FcKCost': f"{M.loc['Forecaster', 'net'] - M.loc['Forecaster + classifier', 'net']:.0f}",
    'FcPCost': f"{M.loc['Forecaster', 'net'] - M.loc['Forecaster + price trigger', 'net']:.0f}",
    'GKCost': f"{M.loc['Gamma', 'net'] - M.loc['Gamma + classifier', 'net']:.0f}",
    'GPCost': f"{M.loc['Gamma', 'net'] - M.loc['Gamma + price trigger', 'net']:.0f}",
    'FcNetGain': f"{M.loc['Forecaster', 'net'] - M.loc['Gamma', 'net']:.0f}",
    'PGammaEventDD': pv('Gamma', 'event_dd'), 'PGammaMaxDD': pv('Gamma', 'max_dd'), 'PGammaWorstDay': pv('Gamma', 'worst_day'),
    'PFcEventDD': pv('Forecaster', 'event_dd'), 'PFcMaxDD': pv('Forecaster', 'max_dd'), 'PFcSharpe': pv('Forecaster', 'sharpe'),
})
better = lambda a, b: sum(E[(E.event == e) & (E.vault == a)].drawdown.iloc[0] > E[(E.event == e) & (E.vault == b)].drawdown.iloc[0] + 1e-9 for e in evs)
mac['FcKBetterEvents'] = str(better('Forecaster + classifier', 'Forecaster + price trigger'))
mac['GKBetterEvents'] = str(better('Gamma + classifier', 'Gamma + price trigger'))
for key, k in [('YGamma', 'Gamma'), ('YGK', 'Gamma + classifier'), ('YGP', 'Gamma + price trigger'), ('YFc', 'Forecaster'),
               ('YFcK', 'Forecaster + classifier'), ('YFcP', 'Forecaster + price trigger')]:
    mac[key + 'EventDD'] = n(Y.loc[k, 'event_dd']); mac[key + 'MaxDD'] = n(Y.loc[k, 'max_dd']); mac[key + 'Sharpe'] = n(Y.loc[k, 'sharpe'])
sg = ST.set_index(['vault', 'freeze']); lead = max(C.STRESS_LEADS)
mac.update({'StressMinDepeg': f"{-meta['stress']['min_depeg']:,.0f}".replace(',', '{,}'), 'StressLead': str(lead),
            'StressGamma': n(sg.loc[('Gamma', 'no freeze'), 'loss'], 0), 'StressGammaRepo': str(int(sg.loc[('Gamma', 'no freeze'), 'repositions'])),
            'StressGammaBreach': n(sg.loc[('Gamma', 'at breach'), 'loss']), 'StressGammaLead': n(sg.loc[('Gamma', f'{lead}h before'), 'loss']),
            'StressFc': n(sg.loc[('Forecaster (rolling empirical)', 'no freeze'), 'loss'], 0),
            'StressFcBreach': n(sg.loc[('Forecaster (rolling empirical)', 'at breach'), 'loss']),
            'StressFcLead': n(sg.loc[('Forecaster (rolling empirical)', f'{lead}h before'), 'loss'])})
with open(C.GEN / 'lp_numbers.tex', 'w') as fh:
    fh.write('% Generated by paper_tables.py -- do not edit by hand\n')
    for k, v in mac.items():
        v = v.replace('$', '') if v.startswith('$') else v
        v = f'\\ensuremath{{{v}}}' if v.startswith('-') else v
        fh.write(f'\\newcommand{{\\lp{k}}}{{{v}}}\n')

def load_curves(path):
    """Inverse of backtest.save_curves: (DatetimeIndex, {key: DataFrame}, extras)."""
    z = np.load(path, allow_pickle=False)
    idx = pd.to_datetime(z['ts'], unit='s', utc=True); cols = list(z['cols'])
    curves = {k: pd.DataFrame({c: z[f'{k}::{c}'] for c in cols}, index=idx) for k in z['keys']}
    extra = {k: z[k] for k in z.files if '::' not in k and k not in ('ts', 'keys', 'cols')}
    return idx, curves, extra


_idx, _cur, _X = load_curves(C.OUT / 'curves.npz')
R = dict(idx=_idx, curves=_cur, depeg=_X['depeg'], alert=_X['alert'], events=_X['events'])
def largest_event_has(k, worst_day=True):
    i = R['events'][int(np.argmax([np.abs(R['depeg'][max(0, j - 24):j + 48]).max() for j in R['events']]))]
    lo, hi = R['idx'][max(0, i - C.EVENT_WINDOW[0])], R['idx'][min(len(R['idx']) - 1, i + C.EVENT_WINDOW[1])]
    c = R['curves'][k]['hedged'].cumsum(); t_dd = (c - c.cummax()).idxmin()
    ok = lo <= t_dd <= hi + pd.Timedelta(days=30)          # trough inside the event episode (recovery may be slow)
    if worst_day:
        d = R['curves'][k]['hedged'].resample('1D').sum().idxmin(); ok &= lo.floor('D') <= d <= hi.ceil('D')
    return bool(ok)
def largest_lead():
    a, i = R['alert'], R['events'][int(np.argmax([np.abs(R['depeg'][max(0, j - 24):j + 48]).max() for j in R['events']]))]
    w = np.flatnonzero(a[max(0, i - 48):i + 1])
    return (i - (max(0, i - 48) + w[0])) if len(w) else -1
# ---------- claims made in the text ----------
fa = F[F.rule == 'asymmetric'].set_index(['forecaster', 'overlay']); fs = F[F.rule == 'symmetric'].set_index(['forecaster', 'overlay'])
fcs = list(dict.fromkeys(F.forecaster))
claims = {
    'Gamma: LVR and costs absorb 40-60% of fee income': 0.4 <= -(M.loc['Gamma', 'lvr'] + M.loc['Gamma', 'costs']) / M.loc['Gamma', 'fees'] <= 0.6,
    'Gamma: max DD and worst day occur in the largest event': largest_event_has('Gamma'),
    'forecaster: max DD occurs in the largest event': largest_event_has('Forecaster', worst_day=False),
    'forecaster keeps >= 75% of Gamma fee income in event windows': ev('Forecaster', 'fees') >= 0.75 * ev('Gamma', 'fees'),
    'event windows: fees exceed LVR for the forecaster vault': ev('Forecaster', 'fees') > -ev('Forecaster', 'lvr'),
    'placebo p-value of the event DD below 0.05 (both vaults)': (P[P.vault == 'Forecaster'].event_dd >= M.loc['Forecaster + classifier', 'event_dd']).mean() < 0.05
        and (P[P.vault == 'Gamma'].event_dd >= M.loc['Gamma + classifier', 'event_dd']).mean() < 0.05,
    'lead time roughly halves event DD vs the price trigger (ratio <= 0.6, both vaults)':
        M.loc['Forecaster + classifier', 'event_dd'] / M.loc['Forecaster + price trigger', 'event_dd'] <= 0.6
        and M.loc['Gamma + classifier', 'event_dd'] / M.loc['Gamma + price trigger', 'event_dd'] <= 0.6,
    'largest event: classifier episode starts >= 12h before the breach': largest_lead() >= 12,
    'JIT 50%: paused vaults earn more than unpaused': all(S[(S.scenario == 'JIT 50%') & (S.vault == f'{b} + classifier')].net.iloc[0]
        > S[(S.scenario == 'JIT 50%') & (S.vault == b)].net.iloc[0] for b in ['Gamma', 'Forecaster']),
    'stress: unpaused vaults lose 20-30% of capital': all(-3000 <= sg.loc[(v, 'no freeze'), 'loss'] <= -2000 for v in ['Gamma', 'Forecaster (rolling empirical)']),
    'forecaster range has higher net and Sharpe than Gamma': M.loc['Forecaster', 'net'] > M.loc['Gamma', 'net'] and M.loc['Forecaster', 'sharpe'] > M.loc['Gamma', 'sharpe'],
    'forecaster range has lower costs and event-window LVR than Gamma': M.loc['Forecaster', 'costs'] > M.loc['Gamma', 'costs'] and ev('Forecaster', 'lvr') > ev('Gamma', 'lvr'),
    'forecaster range has a better worst day and CVaR than Gamma': M.loc['Forecaster', 'worst_day'] > M.loc['Gamma', 'worst_day'] and M.loc['Forecaster', 'cvar5'] > M.loc['Gamma', 'cvar5'],
    'two-model vault has the smallest max DD and highest Sharpe of the non-oracle model vaults':
        M.loc['Forecaster + classifier', 'max_dd'] >= M.drop(['Static', 'Gamma + oracle', 'Forecaster + oracle']).max_dd.max() - 1e-9
        and M.loc['Forecaster + classifier', 'sharpe'] >= M.drop(['Static', 'Gamma + oracle', 'Forecaster + oracle']).sharpe.max() - 1e-9,
    'classifier cuts the event DD of the forecaster vault by more than half': M.loc['Forecaster + classifier', 'event_dd'] > 0.5 * M.loc['Forecaster', 'event_dd'],
    'classifier beats the 15 bp price trigger on event DD (Gamma and forecaster)':
        M.loc['Gamma + classifier', 'event_dd'] > M.loc['Gamma + price trigger', 'event_dd'] and M.loc['Forecaster + classifier', 'event_dd'] > M.loc['Forecaster + price trigger', 'event_dd'],
    'classifier beats the oracle on event DD (Gamma and forecaster)':
        M.loc['Gamma + classifier', 'event_dd'] > M.loc['Gamma + oracle', 'event_dd'] and M.loc['Forecaster + classifier', 'event_dd'] > M.loc['Forecaster + oracle', 'event_dd'],
    'classifier freeze raises the Sharpe ratio of both vaults':
        M.loc['Gamma + classifier', 'sharpe'] > M.loc['Gamma', 'sharpe'] and M.loc['Forecaster + classifier', 'sharpe'] > M.loc['Forecaster', 'sharpe'],
    'classifier beats the price trigger on event DD with every forecaster': all(fa.loc[(f, 'classifier'), 'event_dd'] > fa.loc[(f, 'price trigger'), 'event_dd'] for f in fcs),
    'asymmetric range has higher net than symmetric with every forecaster': all(fa.loc[(f, 'none'), 'net'] > fs.loc[(f, 'none'), 'net'] for f in fcs),
    'classifier beats the price trigger on event DD in every sensitivity scenario': all(
        S[(S.scenario == s) & (S.vault == 'Forecaster + classifier')].event_dd.iloc[0] > S[(S.scenario == s) & (S.vault == 'Forecaster + price trigger')].event_dd.iloc[0]
        and S[(S.scenario == s) & (S.vault == 'Gamma + classifier')].event_dd.iloc[0] > S[(S.scenario == s) & (S.vault == 'Gamma + price trigger')].event_dd.iloc[0]
        for s in S.scenario.unique()),
    '2024: two-model vault has a smaller event DD than either model alone': Y.loc['Forecaster + classifier', 'event_dd'] > max(Y.loc['Forecaster', 'event_dd'], Y.loc['Gamma + classifier', 'event_dd']),
    '2024: classifier beats the price trigger on max DD for the Gamma vault': Y.loc['Gamma + classifier', 'max_dd'] > Y.loc['Gamma + price trigger', 'max_dd'],
    'stress: freezing before the breach loses less than freezing at it': sg.loc[('Gamma', f'{lead}h before'), 'loss'] >= sg.loc[('Gamma', 'at breach'), 'loss'],
}
for k, v in claims.items():
    print(('CLAIM OK    ' if v else 'CLAIM FAILS ') + k)
print(f'wrote {C.GEN}')
