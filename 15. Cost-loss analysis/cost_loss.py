"""Cost-loss evaluation of the early-warning classifier, the probabilistic forecaster and their combination.

No trading simulation: only alerts, forecasts and realised depeg events are used.

    python cost_loss.py      -> results/*.csv

Decision problem (classifier).  Each warning triggers an intervention costing c; each eligible event that is
not anticipated (no alert-episode onset 2-24 h before its first breach, as in the paper's timely recall)
costs L. With D anticipated events out of N and W warnings, the relative economic value, the share of the
saving of a perfect warning system realised by the policy, is
    V(alpha) = (D - alpha W) / (N (1 - alpha)),   alpha = c / L,
relative to never acting (Richardson 2000). V > 0 iff alpha < D / W; V -> recall as alpha -> 0.

Decision problem (forecaster).  At each hour a user protects against |d_{t+h}| >= theta when the forecast
exceedance probability exceeds alpha (the optimal rule for calibrated forecasts); expense per occasion follows
the standard cost-loss model with climatology min(alpha, s) as reference.

Joint.  At each classifier warning, the user acts only if the forecaster's probability of a breach in the next
2-24 hours exceeds alpha (the classifier says when, the forecaster says whether it is worth acting).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / '14. LP vault backtest'))       # shared data and forecast modules
import config as C                                                     # noqa: E402  (folder 14 config: paths)
from data import market, kumo_alert, price_alert, events, panel        # noqa: E402
from forecasts import load_forecasters, es_band, U_SEL                 # noqa: E402
from strategies import episodes_from_flags                             # noqa: E402
from backtest import holdout_window                                    # noqa: E402

OUT = HERE / 'results'
CV_SUMMARY = C.KUMO_RUN / 'five_alert_budget_model_comparison.csv'    # per-candidate CV recall and false alerts
CV_FOLDS = C.KUMO_RUN / 'five_alert_budget_fold_results.csv'           # per-fold alert-episode counts
CV_YEAR = ('2024-01-01 00:00', '2024-12-31 23:00')                     # outer CV test windows cover 2024
SELECTED = ('kumo_tabular', 1.0)
PRECURSORS = (8, 10, 12)                                              # warn when |d| >= theta (price-only rule)
ALPHAS = np.round(np.geomspace(0.002, 0.5, 60), 5)
FC_HORIZONS, FC_THRESHOLDS = (6, 12, 24), (10, 15)
LEAD = (2, 24)


def value(D, W, N, a):
    a = np.asarray(a, float)
    return (D - a * W) / (N * (1 - a))


def onsets(a):
    a = np.asarray(a, bool)
    return np.flatnonzero(a & ~np.r_[False, a[:-1]])


def anticipated(on, ev):
    """Events with an alert-episode onset 2-24 h before the first breach."""
    return int(sum(any(LEAD[0] <= i - o <= LEAD[1] for o in on) for i in ev))


def precursor(m, ev, th):
    on = onsets(price_alert(m, th)); return anticipated(on, ev), len(on)


def classifier_tables():
    S = pd.read_csv(CV_SUMMARY); S = S[S.target_threshold == C.EVENT_BP]
    Fd = pd.read_csv(CV_FOLDS); Fd = Fd[Fd.target_threshold == C.EVENT_BP]
    rows = []
    for _, r in S.iterrows():
        g = Fd[(Fd.model_name == r.model_name) & (Fd.alpha == r.alpha)]
        # warnings = all alert episodes in the outer test windows: false, timely and right-censored
        W = int(g.test_false_alerts.sum() + g.test_timely_alert_episodes.sum() + g.test_right_censored_alert_episodes.sum())
        false_alerts_file = int(g.test_false_alerts.sum())
        rows.append(dict(sample='CV 2024', policy=r.model_name, alpha=r.alpha, family='classifier',
                         selected=(r.model_name, r.alpha) == SELECTED, D=int(r.cv_events_detected), N=int(r.cv_eligible_events),
                         W=W, false=int(r.cv_false_alerts), false_check=false_alerts_file))
    m4 = market(*CV_YEAR); ev4 = events(m4)
    for th in PRECURSORS:
        D, W = precursor(m4, ev4, th)
        rows.append(dict(sample='CV 2024', policy=f'price precursor {th} bp', alpha=np.nan, family='precursor', selected=False,
                         D=D, N=len(ev4), W=W, false=np.nan, false_check=np.nan))
    m = market(*holdout_window()); ev = events(m)
    on = onsets(kumo_alert(m, lag=0))
    rows.append(dict(sample='Holdout', policy='kumo_tabular', alpha=1.0, family='classifier', selected=True,
                     D=anticipated(on, ev), N=len(ev), W=len(on), false=np.nan, false_check=np.nan))
    for th in PRECURSORS:
        D, W = precursor(m, ev, th)
        rows.append(dict(sample='Holdout', policy=f'price precursor {th} bp', alpha=np.nan, family='precursor', selected=False,
                         D=D, N=len(ev), W=W, false=np.nan, false_check=np.nan))
    # per-event anticipation over the holdout (classifier vs precursors)
    pe = pd.DataFrame(dict(event=m.idx[ev], peak_abs_depeg=[np.abs(m.depeg[i:i + 48]).max() for i in ev],
                           classifier=[int(anticipated(on, [i])) for i in ev],
                           **{f'precursor_{th}': [int(anticipated(onsets(price_alert(m, th)), [i])) for i in ev] for th in PRECURSORS}))
    pe.to_csv(OUT / 'holdout_events.csv', index=False)
    T = pd.DataFrame(rows)
    T['recall'] = T.D / T.N; T['breakeven'] = T.D / T.W         # V > 0 iff alpha < D/W
    for a in (0.01, 0.05, 0.1):
        T[f'V_{a}'] = value(T.D, T.W, T.N, a)
    # pooled CV + holdout for the selected classifier and the precursors
    P = []
    for pol in ['kumo_tabular', *[f'price precursor {th} bp' for th in PRECURSORS]]:
        g = T[(T.policy == pol) & ((T.family == 'precursor') | T.selected)]
        P.append(dict(sample='Pooled', policy=pol, family=g.family.iloc[0], selected=pol == 'kumo_tabular', D=g.D.sum(), N=g.N.sum(), W=g.W.sum()))
    P = pd.DataFrame(P); P['recall'] = P.D / P.N; P['breakeven'] = P.D / P.W
    for a in (0.01, 0.05, 0.1):
        P[f'V_{a}'] = value(P.D, P.W, P.N, a)
    k = P[P.policy == 'kumo_tabular'].iloc[0]
    P['crossover_vs_classifier'] = [(k.D - r.D) / (k.W - r.W) if (r.policy != 'kumo_tabular' and k.W > r.W) else np.nan for _, r in P.iterrows()]
    T = pd.concat([T, P], ignore_index=True)
    curves = pd.DataFrame([dict(sample=r['sample'], policy=r.policy, model_alpha=r.get('alpha'), selected=r.selected, family=r.family,
                                cost_loss=a, V=value(r.D, r.W, r.N, a)) for _, r in T.iterrows() for a in ALPHAS])
    return T, curves, (m, ev, on)


def exceed_prob(f, h, thr):
    Q = np.maximum.accumulate(f.Q[:, h - 1, :], axis=1)
    up = 1 - np.array([np.interp(thr, q, U_SEL, left=0, right=1) for q in Q])
    dn = np.array([np.interp(-thr, q, U_SEL, left=0, right=1) for q in Q])
    return np.clip(up + dn, 0, 1)


def rev_prob(p, y, alphas):
    s = y.mean(); out = []
    for a in alphas:
        act = p > a; H = (act & y).sum() / max(y.sum(), 1); F = (act & ~y).sum() / max((~y).sum(), 1)
        E = a * (H * s + F * (1 - s)) + (1 - H) * s
        out.append((min(a, s) - E) / (min(a, s) - a * s))
    return np.array(out)


def forecaster_tables(m, FC):
    d = panel()['depeg_bps']; common = None
    for f in FC.values():
        common = set(f.origin_ts) if common is None else common & set(f.origin_ts)
    common = pd.DatetimeIndex(sorted(common)); common = common[(common >= m.idx[0]) & (common <= m.idx[-1] - pd.Timedelta(hours=25))]
    rows, curves = [], []
    for h in FC_HORIZONS:
        for thr in FC_THRESHOLDS:
            y = (d.reindex(common + pd.Timedelta(hours=h)).abs() >= thr).values
            for nm, f in FC.items():
                pos = pd.Series(np.arange(len(f.origin_ts)), index=f.origin_ts).reindex(common).values
                p = exceed_prob(f, h, thr)[pos]
                bss = 1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2)
                v = rev_prob(p, y, ALPHAS)
                rows.append(dict(h=h, threshold=thr, forecaster=nm, origins=len(y), base_rate=y.mean(), BSS=bss,
                                 **{f'V_{a}': float(rev_prob(p, y, [a])[0]) for a in (0.02, 0.05, 0.1, 0.2)},
                                 positive_share=float(np.mean(v > 0))))
                curves += [dict(h=h, threshold=thr, forecaster=nm, cost_loss=a, V=x) for a, x in zip(ALPHAS, v)]
    return pd.DataFrame(rows), pd.DataFrame(curves)


def breach_prob(f, m, thr=C.EVENT_BP):
    """Forecast probability of |d| >= thr at some horizon 2-24 h ahead (max over horizons of the marginal
    exceedance probability, a lower bound on the path probability), on the score timestamps (no extra lag)."""
    hs = np.arange(LEAD[0] - 1, LEAD[1]); Q = np.maximum.accumulate(f.Q[:, hs, :], axis=2)
    up = 1 - np.apply_along_axis(lambda q: np.interp(thr, q, U_SEL, left=0, right=1), 2, Q)
    dn = np.apply_along_axis(lambda q: np.interp(-thr, q, U_SEL, left=0, right=1), 2, Q)
    return pd.Series((up + dn).max(1), index=f.origin_ts).reindex(m.idx).ffill().bfill().values


def joint_tables(m, ev, on, FC):
    true_on = np.array([any(LEAD[0] <= i - o <= LEAD[1] for i in ev) for o in on])
    screen, curves = [], []
    N, Dk = len(ev), anticipated(on, ev)
    for nm, f in FC.items():
        pi = breach_prob(f, m); el, eh = es_band(f, m, lag=0); esd = np.maximum(m.depeg - el, eh - m.depeg)
        if nm == 'Our model':
            pd.DataFrame(dict(onset=m.idx[on], status=np.where(true_on, 'timely', 'false'), breach_probability=pi[on],
                              es_distance=esd[on])).to_csv(OUT / 'joint_warnings.csv', index=False)
        for q, x in [('breach probability', pi[on]), ('ES distance', esd[on])]:
            t, fl = x[true_on], x[~true_on]
            auc = np.mean([(a > b) + 0.5 * (a == b) for a in t for b in fl])
            screen.append(dict(forecaster=nm, quantity=q, auc=auc, median_timely=np.median(t), median_false=np.median(fl),
                               timely_warnings=int(true_on.sum()), false_warnings=int((~true_on).sum())))
        for a in ALPHAS:
            acted = on[pi[on] > a]                                    # joint: act on warnings the forecaster confirms
            fo = onsets(episodes_from_flags(pi > a, C.KUMO_QUIET))     # forecaster alone as a warning system
            curves += [dict(forecaster=nm, rule='classifier', cost_loss=a, V=value(Dk, len(on), N, a), D=Dk, W=len(on)),
                       dict(forecaster=nm, rule='forecaster', cost_loss=a, V=value(anticipated(fo, ev), len(fo), N, a), D=anticipated(fo, ev), W=len(fo)),
                       dict(forecaster=nm, rule='joint', cost_loss=a, V=value(anticipated(acted, ev), len(acted), N, a), D=anticipated(acted, ev), W=len(acted))]
    return pd.DataFrame(screen), pd.DataFrame(curves)


def main():
    OUT.mkdir(exist_ok=True)
    T, cc, (m, ev, on) = classifier_tables(); T.to_csv(OUT / 'classifier_value.csv', index=False); cc.to_csv(OUT / 'classifier_value_curves.csv', index=False)
    FC = load_forecasters(m)
    F, fc = forecaster_tables(m, FC); F.to_csv(OUT / 'forecaster_value.csv', index=False); fc.to_csv(OUT / 'forecaster_value_curves.csv', index=False)
    S, jc = joint_tables(m, ev, on, FC); S.to_csv(OUT / 'joint_screening.csv', index=False); jc.to_csv(OUT / 'joint_value_curves.csv', index=False)
    pd.set_option('display.width', 220)
    print(T[['sample', 'policy', 'alpha', 'D', 'N', 'W', 'false', 'false_check', 'recall', 'breakeven', 'V_0.01', 'V_0.05', 'V_0.1']].round(3).to_string())
    print(S.round(3).to_string())


if __name__ == '__main__':
    main()
