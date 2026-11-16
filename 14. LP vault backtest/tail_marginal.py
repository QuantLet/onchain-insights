import numpy as np, pandas as pd
from data import panel, OUT
from run_all import load_forecasters
from forecasts import U_SEL
from sklearn.metrics import roc_auc_score
P = panel(); d = P['depeg_bps']; FC = load_forecasters()
rows = []
for thr in [10, 15]:
    for h in [6, 12, 24]:
        for nm, f in FC.items():
            q = np.maximum.accumulate(f.Q[:, h - 1, :], axis=1)
            pl = np.array([np.interp(-thr, qi, U_SEL, left=0.0, right=1.0) for qi in q])
            pu = np.array([1 - np.interp(thr, qi, U_SEL, left=0.0, right=1.0) for qi in q])
            p = np.clip(pl + pu, 0.002, 0.998)
            # target of origin t at horizon h = depeg of bucket t+h
            y = (d.abs() >= thr).astype(float).shift(-h).reindex(f.origin_ts).values
            ok = ~np.isnan(y); p, y = p[ok], y[ok]
            clim = y.mean(); rec = dict(thr=thr, h=h, forecaster=nm, base_rate=clim, mean_p=p.mean(),
                       brier=np.mean((p - y) ** 2), logscore=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                       auc=roc_auc_score(y, p))
            for r in [0.05, 0.1, 0.2]:
                act = p >= r; E_f = np.mean(act * r + (~act) * y); E_c = min(r, clim); E_p = clim * r
                rec[f'V@{r}'] = (E_c - E_f) / (E_c - E_p)
            rows.append(rec)
T = pd.DataFrame(rows); T.to_csv(OUT / 'table7b_tail_marginal.csv')
pd.set_option('display.width', 250); pd.set_option('display.precision', 4)
print(T[T.forecaster.isin(['Our model','TimeXer','TiDE','GARCH','Rolling empirical'])].to_string())
# summary: mean log score rank
S = T[T.forecaster.isin(['Our model','TimeXer','TiDE','GARCH','Rolling empirical'])].copy()
S['rank_log'] = S.groupby(['thr','h'])['logscore'].rank(); S['rank_brier'] = S.groupby(['thr','h'])['brier'].rank()
print(S.groupby('forecaster')[['rank_log','rank_brier']].mean())
