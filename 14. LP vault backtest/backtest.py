"""Run every backtest used in the paper and write results/ (csv, json, pickles).

    python backtest.py            # full run (500 placebo schedules)
    python backtest.py --quick    # 50 placebo schedules
"""
import json, sys, time
import numpy as np
import pandas as pd
import config as C
from data import panel, market, kumo_alert, kumo_cv_alert, price_alert, oracle_alert, events, kumo_scores
from forecasts import load_forecasters, QForecast
from lpsim import Sim, metrics
from strategies import Static, Recenter, ForecastInterval, ForecastSymmetric, Freeze, episode_onsets


def save_curves(path, idx, dfs, cols, **extra):
    """Hourly curves as plain numpy arrays (portable across pandas versions); read with load_curves."""
    arr = {f'{k}::{c}': df[c].values for k, df in dfs.items() for c in cols}
    np.savez_compressed(path, ts=np.asarray((idx - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1s'), dtype=np.int64),
                        keys=np.array(list(dfs)), cols=np.array(cols), **arr, **extra)


def holdout_window():
    """Common coverage of the classifier's holdout scores and the main forecaster's test origins."""
    s = kumo_scores().index
    f = QForecast.from_npz(C.QDIR / f"q_{C.MAIN_FORECASTER.lower().replace(' ', '_')}.npz", C.MAIN_FORECASTER).origin_ts
    start = max(s.min(), f.min()) + pd.Timedelta(hours=2)
    end = min(s.max(), f.max()) + pd.Timedelta(hours=1)
    return start, end


def sim(m, **kw):
    p = dict(V0=C.V0, gas=C.GAS, slip_bps=C.SLIP_BPS, fee_bps=C.POOL_FEE_BPS); p.update(kw)
    return Sim(m, **p)


def vaults(m, f, alert, include_static=True, include_oracle=True):
    """The strategies of the main comparison: Gamma-style vault, forecaster range, each with the classifier's
    freeze and its controls."""
    lo, hi = f.interval(m)
    p15, orc = price_alert(m), oracle_alert(m)
    G = lambda: Recenter(C.GAMMA_HALF_WIDTH)
    F = lambda: ForecastInterval(lo, hi)
    V = {}
    if include_static:
        V['Static'] = lambda: Static(C.STATIC_HALF_WIDTH)
    V.update({'Gamma': G, 'Gamma + classifier': lambda: Freeze(G(), alert), 'Gamma + price trigger': lambda: Freeze(G(), p15)})
    if include_oracle:
        V['Gamma + oracle'] = lambda: Freeze(G(), orc)
    V.update({'Forecaster': F, 'Forecaster + classifier': lambda: Freeze(F(), alert),
              'Forecaster + price trigger': lambda: Freeze(F(), p15)})
    if include_oracle:
        V['Forecaster + oracle'] = lambda: Freeze(F(), orc)
    return V


def run_set(s, m, V, ev):
    dfs = {k: s.run(mk()) for k, mk in V.items()}
    T = pd.DataFrame([metrics(df, m, ev, C.EVENT_WINDOW, label=k) for k, df in dfs.items()])
    return T, dfs


def per_event(m, dfs, ev, keys):
    rows = []
    for i in ev:
        sl = slice(max(0, i - C.EVENT_WINDOW[0]), i + C.EVENT_WINDOW[1])
        for k in keys:
            d = dfs[k].iloc[sl]; c = d['hedged'].cumsum().values / C.V0 * 1e4
            rows.append(dict(event=str(m.idx[i])[:16], vault=k, drawdown=(c - np.maximum.accumulate(np.r_[0.0, c])[1:]).min(),
                             fees=d.fees.sum() / C.V0 * 1e4, lvr=d.lvr.sum() / C.V0 * 1e4,
                             costs=-(d.gas.sum() + d.swapcost.sum()) / C.V0 * 1e4, max_abs_depeg=np.abs(m.depeg[sl]).max()))
    return pd.DataFrame(rows)


def regimes(m, dfs, ev, alert):
    win = np.zeros(m.n, bool)
    for i in ev:
        win[max(0, i - C.EVENT_WINDOW[0]):i + C.EVENT_WINDOW[1]] = True
    rows = []
    for k, df in dfs.items():
        for r, msk in [('event windows', win), ('other alert hours', alert & ~win), ('calm hours', ~win & ~alert)]:
            d = df[msk]
            rows.append(dict(vault=k, regime=r, hours=int(msk.sum()), fees=d.fees.sum() / C.V0 * 1e4, lvr=d.lvr.sum() / C.V0 * 1e4,
                             costs=-(d.gas.sum() + d.swapcost.sum()) / C.V0 * 1e4, net=d.hedged.sum() / C.V0 * 1e4))
    return pd.DataFrame(rows)


def placebo(s, m, ev, alert, base, n):
    """Classifier episodes (with their durations) placed at random dates."""
    rng = np.random.default_rng(C.SEED)
    on = episode_onsets(alert); off = np.flatnonzero(alert & ~np.r_[alert[1:], False]); dur = off - on + 1
    rows = []
    for _ in range(n):
        a = np.zeros(m.n, bool)
        for d in dur:
            s0 = rng.integers(0, m.n - d); a[s0:s0 + d] = True
        rows.append(metrics(s.run(Freeze(base(), a)), m, ev, C.EVENT_WINDOW))
    return pd.DataFrame(rows)


def alert_stats(m, alert, ev):
    on = episode_onsets(alert)
    leads = []
    for i in ev:
        w = np.flatnonzero(alert[max(0, i - 24):i + 1])
        if len(w):
            j = max(0, i - 24) + w[0]
            while j > 0 and alert[j - 1]:
                j -= 1
            leads.append(i - j)
    false = sum(1 for o in on if not np.any((ev >= o) & (ev <= o + 24 + C.KUMO_CV_EPISODE_H)))
    return dict(alert_share=float(alert.mean()), episodes=int(len(on)), false_episodes=int(false),
                events=int(len(ev)), detected=int(len(leads)), median_lead_from_onset=float(np.median(leads)) if leads else None)


def main(n_placebo):
    t0 = time.time(); C.OUT.mkdir(exist_ok=True); meta = {}
    # ---------------- holdout ----------------
    start, end = holdout_window(); m = market(start, end); s = sim(m)
    ka, ev = kumo_alert(m), events(m); FC = load_forecasters(m); f = FC[C.MAIN_FORECASTER]
    meta['holdout'] = dict(start=str(start), end=str(end), years=m.years, max_abs_depeg=float(np.abs(m.depeg).max()),
                           classifier=alert_stats(m, ka, ev), price_trigger_share=float(price_alert(m).mean()))
    V = vaults(m, f, ka)
    T, dfs = run_set(s, m, V, ev); T.to_csv(C.OUT / 'main.csv')
    per_event(m, dfs, ev, list(V)).to_csv(C.OUT / 'per_event.csv', index=False)
    regimes(m, dfs, ev, ka).to_csv(C.OUT / 'regimes.csv', index=False)
    save_curves(C.OUT / 'curves.npz', m.idx, dfs, ['hedged', 'fees', 'lvr', 'gas', 'swapcost'],
                depeg=m.depeg, alert=ka, price_trigger=price_alert(m), events=ev)
    print(T[['net', 'sharpe', 'event_dd', 'max_dd', 'worst_day']].round(2).to_string())
    # placebo
    lo, hi = f.interval(m)
    P = []
    for lab, base in [('Gamma', lambda: Recenter(C.GAMMA_HALF_WIDTH)), ('Forecaster', lambda: ForecastInterval(lo, hi))]:
        d = placebo(s, m, ev, ka, base, n_placebo); d['vault'] = lab; P.append(d)
    pd.concat(P).to_csv(C.OUT / 'placebo.csv', index=False)
    # every forecaster, asymmetric (main rule) and symmetric (robustness)
    rows = []
    for nm, fc in FC.items():
        lo, hi = fc.interval(m); med = fc.median(m)
        for rule, mk in [('asymmetric', lambda: ForecastInterval(lo, hi)), ('symmetric', lambda: ForecastSymmetric(lo, hi, med))]:
            for ov, a in [('none', None), ('classifier', ka), ('price trigger', price_alert(m))]:
                df = s.run(mk() if a is None else Freeze(mk(), a))
                r = metrics(df, m, ev, C.EVENT_WINDOW); r['forecaster'], r['rule'], r['overlay'] = nm, rule, ov; rows.append(r)
    pd.DataFrame(rows).to_csv(C.OUT / 'forecasters.csv', index=False)
    # sensitivity
    rows = []
    scen = [('Base', {}), ('Fees reinvested', dict(compound='rebalance')), ('Low cost', dict(gas=2.0, slip_bps=0.0)),
            ('High cost', dict(gas=50.0, slip_bps=1.0)), ('100k vault', dict(V0=1e5)), ('5M vault', dict(V0=5e6)), ('JIT 50%', dict(jit_haircut=0.5))]
    Vs = vaults(m, f, ka, include_static=False, include_oracle=False)
    for tag, kw in scen:
        ss = sim(m, **kw)
        for k, mk in Vs.items():
            r = metrics(ss.run(mk()), m, ev, C.EVENT_WINDOW); r['scenario'], r['vault'] = tag, k; rows.append(r)
    pd.DataFrame(rows).to_csv(C.OUT / 'sensitivity.csv', index=False)
    # ---------------- 2024 (classifier CV folds; rolling empirical forecaster) ----------------
    m4 = market(*C.Y2024); k4, ev4 = kumo_cv_alert(m4), events(m4)
    f4 = QForecast.rolling_empirical(m4.idx[0] - pd.Timedelta(hours=2), m4.idx[-1] + pd.Timedelta(hours=1))
    T4, _ = run_set(sim(m4), m4, vaults(m4, f4, k4, include_static=True), ev4); T4.to_csv(C.OUT / 'y2024.csv')
    meta['y2024'] = dict(start=C.Y2024[0], end=C.Y2024[1], classifier=alert_stats(m4, k4, ev4))
    # ---------------- stress scenario: March 2023 (losses only, no fee data) ----------------
    ms = market(*C.STRESS_WINDOW); s3 = sim(ms); first = int(np.argmax(np.abs(ms.depeg) >= C.EVENT_BP))
    fr = QForecast.rolling_empirical(ms.idx[0] - pd.Timedelta(hours=2), ms.idx[-1] + pd.Timedelta(hours=1)); lo3, hi3 = fr.interval(ms)
    p3 = price_alert(ms)
    freezes = {'no freeze': None, 'at breach': p3}
    for L in C.STRESS_LEADS:
        a = p3.copy(); a[max(0, first - L):first + 1] = True; freezes[f'{L}h before'] = a
    rows, sc = [], {}
    w = slice(*C.STRESS_SUM)
    for lab, mk in [('Gamma', lambda: Recenter(C.GAMMA_HALF_WIDTH)), ('Forecaster (rolling empirical)', lambda: ForecastInterval(lo3, hi3))]:
        for fz, a in freezes.items():
            df = s3.run(mk() if a is None else Freeze(mk(), a)); d = df.loc[w]
            loss = (d.lvr.sum() - d.gas.sum() - d.swapcost.sum()) / C.V0 * 1e4
            rows.append(dict(vault=lab, freeze=fz, loss=loss, lvr=d.lvr.sum() / C.V0 * 1e4, repositions=int(d.reb.sum())))
            sc[(lab, fz)] = df[['lvr', 'gas', 'swapcost']]
    d = s3.run(Static(C.STATIC_HALF_WIDTH)).loc[w]
    rows.append(dict(vault='Static', freeze='no freeze', loss=(d.lvr.sum() - d.gas.sum() - d.swapcost.sum()) / C.V0 * 1e4,
                     lvr=d.lvr.sum() / C.V0 * 1e4, repositions=int(d.reb.sum())))
    pd.DataFrame(rows).to_csv(C.OUT / 'stress.csv', index=False)
    save_curves(C.OUT / 'stress_curves.npz', ms.idx, {f'{a}|{b}': d for (a, b), d in sc.items()}, ['lvr', 'gas', 'swapcost'],
                depeg=ms.depeg, first=np.array(first))
    meta['stress'] = dict(first_breach=str(ms.idx[first]), min_depeg=float(ms.depeg.min()))
    meta['n_placebo'] = n_placebo
    json.dump(meta, open(C.OUT / 'meta.json', 'w'), indent=1, default=str)
    print(f'done in {time.time() - t0:.0f}s -> {C.OUT}')


if __name__ == '__main__':
    main(50 if '--quick' in sys.argv else C.N_PLACEBO)
