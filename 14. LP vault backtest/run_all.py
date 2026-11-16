"""Economic backtest of LP vault strategies driven by the two-model framework.

    python "14. LP vault backtest/run_all.py"            # full run (tables + figures)
    python "14. LP vault backtest/run_all.py" --quick    # fewer placebo draws

Outputs go to 14. LP vault backtest/results/.
"""
import sys
import numpy as np
import pandas as pd

from data import (make_market, kumo_alert, price_alert, oracle_alert, event_mask, events,
                  kumo_cv_alert_2024, QDIR, OUT, HERE, panel)
from lpsim import Sim, Market, metrics
from strategies import (Hold, Static, Recenter, VolRecenter, Gated, ForecastWidth, Combined,
                        episode_onsets)
from forecasts import QForecast

QUICK = '--quick' in sys.argv
N_PLACEBO = 100 if QUICK else 500
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

COLS = ['fee_apr', 'lvr_apr', 'cost_apr', 'net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps',
        'worst_day_bps', 'hedged_cvar5_daily_bps', 'rebal_per_month', 'avg_width_ticks',
        'event_hedged_bps', 'calm_hedged_apr']
NICE = {'fee_apr': 'Fees (bp/yr)', 'lvr_apr': 'LVR (bp/yr)', 'cost_apr': 'Costs (bp/yr)',
        'net_vs_rebal_apr': 'Net vs reb. (bp/yr)', 'hedged_sharpe': 'Sharpe', 'hedged_sortino': 'Sortino (hourly)',
        'hedged_maxdd_bps': 'Max DD (bp)', 'worst_day_bps': 'Worst day (bp)',
        'hedged_cvar5_daily_bps': 'CVaR5% day (bp)', 'rebal_per_month': 'Rebal./month',
        'avg_width_ticks': 'Width (ticks)', 'event_hedged_bps': 'Event windows (bp)',
        'calm_hedged_apr': 'Calm (bp/yr)'}


# ----------------------------------------------------------------------------------------
def load_forecasters():
    fc = {}
    names = [('ours', 'Our model'), ('timexer', 'TimeXer'), ('tide', 'TiDE'), ('garch', 'GARCH'),
             ('arima', 'ARIMA'), ('naive', 'Naive')]
    for key, label in names:
        p = QDIR / f'q_{key}.npz'
        if p.exists():
            fc[label] = QForecast.from_npz(p, label)
    cache = HERE / '.rolling_empirical.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        fc['Rolling empirical'] = QForecast(z['Q'], 'Rolling empirical', pd.to_datetime(z['ts'], unit='s', utc=True))
    else:
        f = QForecast.rolling_empirical()
        np.savez_compressed(cache, Q=f.Q, ts=np.asarray((f.origin_ts - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1s'), dtype=np.int64))
        fc['Rolling empirical'] = f
    return fc


def width_strategy(f, m, alpha=0.2, H=6, name=None):
    lo, hi = f.interval(m, alpha, H)
    med = f.median_path(m, H)
    return ForecastWidth(lo, hi, med, name=name or f'{f.name} width q{alpha/2:.2f}-{1-alpha/2:.2f} H{H}')


def run(sim, m, s, em, label=None):
    df = sim.run(s)
    return df, metrics(df, m, label=label or s.name, event_mask=em)


def stationary_bootstrap_diff(a, b, B=2000, block=7, seed=0):
    """Mean daily difference a-b with a stationary bootstrap CI and one-sided p (H0: mean<=0)."""
    d = (a - b).values; n = len(d); rng = np.random.default_rng(seed)
    p = 1.0 / block; means = np.empty(B)
    for r in range(B):
        idx = np.empty(n, int); i = rng.integers(n)
        for t in range(n):
            if t == 0 or rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
            idx[t] = i
        means[r] = d[idx].mean()
    mu = d.mean()
    # one-sided p-value for H0: E[a-b] <= 0 (bootstrap distribution recentred under the null)
    return mu, np.quantile(means, 0.025), np.quantile(means, 0.975), float(((means - mu) >= mu).mean())


def to_latex(df, path, cols, caption, label):
    t = df[cols].rename(columns=NICE).copy()
    fmt = {c: (lambda x: f'{x:,.1f}') for c in t.columns}
    with open(path, 'w') as fh:
        tex = t.to_latex(formatters=fmt, escape=True, caption=caption, label=label,
                         column_format='l' + 'r' * len(t.columns))
        fh.write(tex.replace('±', '$\\pm$').replace('≥', '$\\geq$').replace('NaN', '--')
                    .replace('|d|', '$|d|$').replace('-0.0 ', '0.0 '))


# ----------------------------------------------------------------------------------------
def main():
    m = make_market(); sim = Sim(m); em = event_mask(m)
    ka = kumo_alert(m); p8 = price_alert(m, 8); orc = oracle_alert(m)
    FC = load_forecasters()
    main_fc = 'Our model' if 'Our model' in FC else 'TiDE'
    print('window', m.idx[0], '->', m.idx[-1], f'({m.n} h); events:', [str(t) for t in events(m)])
    print('forecasters:', list(FC), '| main forecaster:', main_fc)
    print(f'Kumo alert share {ka.mean():.3f} ({len(episode_onsets(ka))} episodes); '
          f'price |d|>=8 share {p8.mean():.3f} ({len(episode_onsets(p8))} episodes)')

    # ---------------- Table 1: main comparison ----------------
    rows, curves = [], {}
    spec = [('Standard', Hold()), ('Standard', Static(5)), ('Standard', Recenter(2)),
            ('Standard', VolRecenter(2)),
            ('Classifier', Gated(Recenter(2), ka, 'freeze', name='Recenter ±2 + Kumo alert freeze')),
            ('Classifier', Gated(Recenter(2), ka, 'wide', W=15, name='Recenter ±2 + Kumo alert parity ±15')),
            ('Control', Gated(Recenter(2), p8, 'freeze', name='Recenter ±2 + price |d|≥8 freeze')),
            ('Control', Gated(Recenter(2), orc, 'freeze', name='Recenter ±2 + oracle freeze'))]
    for nm, f in FC.items():
        spec.append(('Forecaster', width_strategy(f, m, name=f'{nm} forecast width')))
    for nm in [main_fc] + [x for x in ['TiDE'] if x != main_fc and x in FC]:
        spec.append(('Two-model', Gated(width_strategy(FC[nm], m), ka, 'freeze',
                                        name=f'{nm} forecast width + Kumo alert freeze')))
        lt, ht = FC[nm].interval(m, 0.02, 24)
        spec.append(('Two-model', Combined(width_strategy(FC[nm], m), ka, lt, ht,
                                           name=f'{nm} forecast width + Kumo tail-range')))
    # ES-based strategies (forecast tails): asymmetric ES bands and the classifier gate filtered by forecast ES
    from es_strategies import ESBand, es_interval, es_tail_distance
    fes = FC[main_fc]
    lo5, hi5, med6 = es_interval(fes, m, 0.05, 6)
    g_es = ka & (es_tail_distance(fes, m) > 10)
    spec += [('Forecaster ES', ESBand(lo5, hi5, med6, 0.5, name=f'{main_fc} ES band')),
             ('Two-model ES', Gated(width_strategy(fes, m), g_es, 'freeze', name=f'{main_fc} width + Kumo∧ES gate')),
             ('Two-model ES', Gated(ESBand(lo5, hi5, med6, 0.5), g_es, 'freeze', name=f'{main_fc} ES band + Kumo∧ES gate')),
             ('Classifier', Gated(Recenter(2), g_es, 'freeze', name='Recenter ±2 + Kumo∧ES gate'))]
    for fam, s in spec:
        df, r = run(sim, m, s, em); r['family'] = fam; rows.append(r); curves[s.name] = df
    T1 = pd.DataFrame(rows)
    print('\nTABLE 1 - holdout, $1M, gas $10/tx, 1.5bp swap cost\n', T1[COLS].round(2).to_string())
    T1.to_csv(OUT / 'table1_main.csv')
    to_latex(T1, OUT / 'table1_main.tex', ['fee_apr', 'lvr_apr', 'cost_apr', 'net_vs_rebal_apr',
             'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps', 'rebal_per_month', 'event_hedged_bps'],
             'LP vault backtest on the USDC--USDT 0.01\\% Uniswap V3 pool, holdout 2025-01-15 to 2026-05-04. '
             'PnL is measured against a delta-matched rebalancing portfolio (fees $-$ LVR $-$ costs), '
             'in basis points of a \\$1M vault.', 'tab:lp_backtest')
    pd.to_pickle(curves, OUT / 'curves.pkl')

    # ---------------- Table 2: placebo tests ----------------
    rng = np.random.default_rng(7)
    ons = episode_onsets(ka); ends = np.flatnonzero(ka & ~np.r_[ka[1:], False]); durs = ends - ons + 1

    def placebo():
        a = np.zeros(m.n, bool)
        for d in durs:
            s0 = rng.integers(0, m.n - d); a[s0:s0 + d] = True
        return a
    T2 = []
    for label, mk in [('Recenter ±2 + alert freeze', lambda a: Gated(Recenter(2), a, 'freeze')),
                      (f'{main_fc} width + alert freeze', lambda a: Gated(width_strategy(FC[main_fc], m), a, 'freeze'))]:
        real = metrics(sim.run(mk(ka)), m, event_mask=em)
        P = pd.DataFrame([metrics(sim.run(mk(placebo())), m, event_mask=em) for _ in range(N_PLACEBO)])
        for c in ['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_maxdd_bps', 'worst_day_bps',
                  'hedged_cvar5_daily_bps', 'event_hedged_bps']:
            T2.append(dict(strategy=label, metric=NICE[c], kumo=real[c], placebo_median=P[c].median(),
                           placebo_p05=P[c].quantile(0.05), placebo_p95=P[c].quantile(0.95),
                           p_value=(P[c] >= real[c]).mean()))
    T2 = pd.DataFrame(T2)
    print(f'\nTABLE 2 - placebo ({N_PLACEBO} random alert schedules, same episode count/durations)\n',
          T2.round(3).to_string())
    T2.to_csv(OUT / 'table2_placebo.csv')

    # ---------------- Table 3: forecaster horse race (decision-based evaluation) ----------------
    T3 = []
    daily = {}
    for nm, f in FC.items():
        for alpha, H in [(0.5, 6), (0.2, 6), (0.2, 24), (0.1, 24)]:
            s = width_strategy(f, m, alpha, H)
            df, r = run(sim, m, s, em)
            r['forecaster'] = nm; r['interval'] = f'q{alpha/2:.2f}-q{1-alpha/2:.2f}'; r['H'] = H
            lo, hi = f.interval(m, alpha, H)
            r['coverage_next_h'] = np.mean((m.depeg[1:] >= lo[:-1]) & (m.depeg[1:] <= hi[:-1]))
            T3.append(r)
            if (alpha, H) == (0.2, 6):
                daily[nm] = df['hedged'].resample('1D').sum() / 1e6 * 1e4
    T3 = pd.DataFrame(T3)
    print('\nTABLE 3 - forecaster horse race, identical width rule\n',
          T3[['forecaster', 'interval', 'H', 'coverage_next_h', 'net_vs_rebal_apr', 'hedged_sharpe',
              'hedged_maxdd_bps', 'worst_day_bps', 'rebal_per_month']].round(2).to_string())
    T3.to_csv(OUT / 'table3_forecasters.csv')
    vs = curves['Vol-scaled recenter (x2)']['hedged'].resample('1D').sum() / 1e6 * 1e4
    rb = curves['Recenter ±2']['hedged'].resample('1D').sum() / 1e6 * 1e4
    BT = []
    for nm, d in daily.items():
        for ref_name, ref in [('Vol-scaled recenter (x2)', vs), ('Recenter ±2', rb)] + \
                [(o, daily[o]) for o in daily if o != nm and nm == main_fc]:
            mu, lo, hi, p = stationary_bootstrap_diff(d, ref, B=500 if QUICK else 2000)
            BT.append(dict(strategy=nm, benchmark=ref_name, mean_daily_diff_bp=mu, ci_lo=lo, ci_hi=hi,
                           annualised_bp=mu * 365, p_one_sided=p))
    BT = pd.DataFrame(BT)
    print('\nBootstrap (stationary, 7-day mean block) of daily hedged PnL differences\n', BT.round(4).to_string())
    BT.to_csv(OUT / 'table3b_bootstrap.csv')

    # ---------------- Table 4: 2024 CV period (classifier only) ----------------
    m24 = Market(panel(), '2024-01-01 00:00', '2024-12-31 23:00'); s24 = Sim(m24); em24 = event_mask(m24)
    k24 = kumo_cv_alert_2024(m24); p24 = price_alert(m24, 8); o24 = oracle_alert(m24)
    T4 = []
    for base in [lambda: Static(5), lambda: Recenter(2), lambda: VolRecenter(2)]:
        b = base(); T4.append(metrics(s24.run(b), m24, label=b.name, event_mask=em24))
        if isinstance(b, Static):
            continue
        for nm, al in [('Kumo CV alerts', k24), ('price |d|≥8', p24), ('oracle', o24)]:
            T4.append(metrics(s24.run(Gated(base(), al, 'freeze')), m24, label=f'{b.name} + {nm} freeze', event_mask=em24))
    T4 = pd.DataFrame(T4)
    print(f'\nTABLE 4 - 2024 (CV test folds, {len(events(m24))} events)\n', T4[COLS].round(2).to_string())
    T4.to_csv(OUT / 'table4_2024.csv')

    # ---------------- Table 5: sensitivity ----------------
    T5 = []
    keys = [lambda: Recenter(2), lambda: Gated(Recenter(2), ka, 'freeze', name='Recenter ±2 + Kumo freeze'),
            lambda: Gated(Recenter(2), p8, 'freeze', name='Recenter ±2 + price freeze'),
            lambda: width_strategy(FC[main_fc], m, name=f'{main_fc} width'),
            lambda: Gated(width_strategy(FC[main_fc], m), ka, 'freeze', name=f'{main_fc} width + Kumo freeze')]
    for tag, kw in [('base', {}), ('low cost (gas $2, no slippage)', dict(gas=2, slip_bps=0)),
                    ('high cost (gas $50, 1bp slippage)', dict(gas=50, slip_bps=1.0)),
                    ('$100k vault', dict(V0=1e5)), ('$5M vault', dict(V0=5e6)),
                    ('50% of fees to JIT', dict(jit_haircut=0.5))]:
        ss = Sim(m, **kw)
        for k in keys:
            s = k(); r = metrics(ss.run(s), m, label=s.name, event_mask=em); r['scenario'] = tag; T5.append(r)
    T5 = pd.DataFrame(T5)
    print('\nTABLE 5 - sensitivity\n', T5[['scenario', 'net_vs_rebal_apr', 'hedged_sharpe', 'hedged_maxdd_bps',
                                          'worst_day_bps']].round(2).to_string())
    T5.to_csv(OUT / 'table5_sensitivity.csv')

    figures(m, curves, T1, ka, main_fc)


# ----------------------------------------------------------------------------------------
def figures(m, curves, T1, ka, main_fc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e7e6e2', 'grid.linewidth': 0.6,
                         'legend.frameon': False})
    sel = [('Static parity ±5', 'Static parity ±5'), ('Recenter ±2', 'Gamma-style recenter ±2'),
           ('Recenter ±2 + Kumo alert freeze', 'Recenter ±2 + Kumo gate'),
           (f'{main_fc} forecast width + Kumo alert freeze', f'{main_fc} forecast width + Kumo gate')]
    ev = events(m)

    # Fig 1: cumulative hedged PnL
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (k, lab) in enumerate(sel):
        y = curves[k]['hedged'].cumsum() / 1e6 * 1e4
        ax.plot(y.index, y.values, lw=1.6, color=C[i], label=lab)
        ax.annotate(lab, (y.index[-1], y.values[-1]), xytext=(4, 0), textcoords='offset points',
                    fontsize=7.5, color='#52514e', va='center')
    for t in ev:
        ax.axvline(t, color='#c3c2b7', lw=0.8, ls=':', zorder=0)
    ax.set_ylabel('Cumulative PnL vs rebalancing (bp of capital)')
    ax.set_title('LP vault PnL net of LVR and costs, holdout (dotted: 15 bp depeg events)', fontsize=9, loc='left')
    ax.legend(loc='upper left', fontsize=7.5)
    ax.set_xlim(y.index[0], y.index[-1] + pd.Timedelta(days=80))
    fig.tight_layout(); fig.savefig(OUT / 'fig1_cumulative_pnl.png', dpi=200); fig.savefig(OUT / 'fig1_cumulative_pnl.pdf')
    plt.close(fig)

    # Fig 2: event zoom (2026-02-04 event)
    t0, t1 = pd.Timestamp('2026-02-03 00:00', tz='UTC'), pd.Timestamp('2026-02-07 00:00', tz='UTC')
    win = (m.idx >= t0) & (m.idx <= t1); idx = m.idx[win]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={'height_ratios': [1.3, 1.3, 1]})
    kas = pd.Series(ka, index=m.idx)[win]
    for ax, (k, lab), col in [(axs[0], sel[1], C[1]), (axs[1], sel[3], C[3])]:
        df = curves[k].loc[win]
        ax.fill_between(idx, df['ta'], df['tb'], step='post', color=col, alpha=0.25, lw=0, label=f'{lab}: LP range')
        ax.plot(idx, m.depeg[win], color='#0b0b0b', lw=1.2, label='Pool depeg (bp)')
        for a, b in _spans(kas):
            ax.axvspan(a, b, color='#e7e6e2', zorder=0)
        ax.set_ylabel('bp from parity'); ax.legend(loc='upper left', fontsize=7.5)
        ax.set_ylim(-2, 34)
    axs[0].set_title('Feb 2026 event: LP ranges vs pool price (grey: Kumo alert episodes, 1h lag)', fontsize=9, loc='left')
    for i, (k, lab) in enumerate(sel[1:]):
        y = curves[k]['hedged'].loc[win].cumsum() / 1e6 * 1e4
        axs[2].plot(idx, y, lw=1.6, color=C[i + 1], label=lab)
    axs[2].set_ylabel('Cum. PnL vs reb. (bp)'); axs[2].legend(loc='lower left', fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_event_zoom.png', dpi=200); fig.savefig(OUT / 'fig2_event_zoom.pdf')
    plt.close(fig)

    # Fig 3: risk-return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fam_col = {'Standard': C[0], 'Classifier': C[1], 'Control': '#8a8984', 'Forecaster': C[2], 'Two-model': C[6],
               'Forecaster ES': C[5], 'Two-model ES': C[3]}
    for fam, g in T1[T1.index != 'HODL 50/50'].groupby('family'):
        ax.scatter(-g['hedged_maxdd_bps'], g['net_vs_rebal_apr'], s=36, color=fam_col[fam], label=fam,
                   edgecolor='white', linewidth=1.5, zorder=3)
        for nm, r in g.iterrows():
            lab, off = _short(nm)
            if lab is None:
                continue
            ax.annotate(lab, (-r['hedged_maxdd_bps'], r['net_vs_rebal_apr']), xytext=off,
                        textcoords='offset points', fontsize=6.5, color='#52514e')
    ax.set_xscale('log'); ax.set_xlabel('Max drawdown of hedged PnL (bp, log scale)')
    ax.set_ylabel('Net PnL vs rebalancing (bp/yr)')
    ax.set_title('Return vs drawdown, holdout', fontsize=9, loc='left'); ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT / 'fig3_risk_return.png', dpi=200); fig.savefig(OUT / 'fig3_risk_return.pdf')
    plt.close(fig)
    print('figures written to', OUT)


def _short(nm):
    table = {
        'Static parity ±5': ('Static ±5', (5, -3)),
        'Recenter ±2': ('Recenter ±2', (5, -3)),
        'Vol-scaled recenter (x2)': ('Vol-scaled', (5, -10)),
        'Recenter ±2 + Kumo alert freeze': ('Recenter ±2 + Kumo', (5, 2)),
        'Recenter ±2 + Kumo alert parity ±15': (None, (0, 0)),
        'Recenter ±2 + price |d|≥8 freeze': ('+ price trigger', (-62, -9)),
        'Recenter ±2 + oracle freeze': ('+ oracle', (-30, 5)),
    }
    if nm in table:
        return table[nm]
    if 'tail-range' in nm:
        return (None, (0, 0))
    if 'Kumo alert freeze' in nm:
        return (nm.split(' forecast')[0] + ' width + Kumo', (5, 3))
    return (nm.replace(' forecast width', ' width'), (5, -3))


def _spans(s):
    out = []; a = None
    for t, v in s.items():
        if v and a is None:
            a = t
        if not v and a is not None:
            out.append((a, t)); a = None
    if a is not None:
        out.append((a, s.index[-1]))
    return out


if __name__ == '__main__':
    main()
