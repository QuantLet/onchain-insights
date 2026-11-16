"""Tail decision test: P(breach of |d|>=thr within the next 24h) from each forecaster's marginal quantiles.
Scores: Brier, log score, AUC; economic: relative value of a cost-loss protection rule (Richardson, 2000)."""
import numpy as np, pandas as pd
from data import panel, OUT
from run_all import load_forecasters
from forecasts import U_SEL
P = panel(); d = P['depeg_bps']
FC = load_forecasters()
lag = 1  # signal from origin k-1 usable at k
def exceed_prob(f, thr, hs):
    Q = f.Q[:, hs, :]  # N,H,L ascending in u
    N, H, L = Q.shape
    # P(d <= -thr) and P(d >= thr) per horizon via interpolation of the CDF on the quantile grid
    pl = np.empty((N, H)); pu = np.empty((N, H))
    for h in range(H):
        q = Q[:, h, :]
        q = np.maximum.accumulate(q, axis=1)
        pl[:, h] = np.array([np.interp(-thr, q[i], U_SEL, left=0.0, right=1.0) for i in range(N)])
        pu[:, h] = np.array([1 - np.interp(thr, q[i], U_SEL, left=0.0, right=1.0) for i in range(N)])
    # clip to the grid's resolution: below the smallest level we only know p < u_min
    p = np.clip(np.maximum(pl, pu).max(1), U_SEL[0] / 2, 1 - U_SEL[0] / 2)
    return pd.Series(p, index=f.origin_ts)
rows = []; probs = {}
for thr in [10, 15]:
    # outcome for origin t: any |d| >= thr in buckets t+2 .. t+25 (horizons h=2..25 after the 1h lag)
    a = (d.abs() >= thr).astype(int)
    fut = sum(a.shift(-h) for h in range(2, 26))
    y = (fut > 0).astype(float)
    hs = np.arange(1, 24)
    base = None
    for nm, f in FC.items():
        p = exceed_prob(f, thr, hs)
        yy = y.reindex(p.index).dropna(); p = p.reindex(yy.index)
        probs[(thr, nm)] = p
        brier = np.mean((p - yy) ** 2); ll = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
        from sklearn.metrics import roc_auc_score, average_precision_score
        rec = dict(thr=thr, forecaster=nm, base_rate=yy.mean(), mean_p=p.mean(), brier=brier, logscore=ll,
                   auc=roc_auc_score(yy, p), auprc=average_precision_score(yy, p))
        # relative economic value at cost/loss ratios
        clim = yy.mean()
        for r in [0.02, 0.05, 0.1, 0.2, 0.3]:
            act = p >= r
            E_f = np.mean(act * r + (~act) * yy * 1.0)        # expense per unit loss
            E_c = min(r, clim)                                 # climatology: always or never protect
            E_p = clim * r                                     # perfect information
            rec[f'V@{r}'] = (E_c - E_f) / (E_c - E_p)
        rows.append(rec)
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.precision', 4)
print(T.to_string())
T.to_csv(OUT / 'table7_tail_value.csv')
pd.to_pickle(probs, OUT / 'tail_probs.pkl')
