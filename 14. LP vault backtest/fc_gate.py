"""Forecaster-as-gate: alert when P(|d_{t+h}|>=10bp) from the forecaster exceeds a cutoff set so the alert-hour
share matches Kumo's (11.6%), same 12h quiet reset and 1h lag. Compare with Kumo and the price trigger."""
import numpy as np, pandas as pd
from data import panel, make_market, event_mask, kumo_alert, price_alert, OUT
from lpsim import Sim, metrics
from strategies import Recenter, Gated, episodes_from_flags, episode_onsets
from run_all import load_forecasters
from forecasts import U_SEL
P = panel(); m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m); p8 = price_alert(m, 8)
FC = load_forecasters(); target = ka.mean()
def tailprob(f, thr=10, hs=(6, 12)):
    out = 0
    for h in hs:
        q = np.maximum.accumulate(f.Q[:, h - 1, :], axis=1)
        pl = np.array([np.interp(-thr, qi, U_SEL, left=0.0, right=1.0) for qi in q])
        pu = np.array([1 - np.interp(thr, qi, U_SEL, left=0.0, right=1.0) for qi in q])
        out = np.maximum(out, pl + pu)
    return pd.Series(out, index=f.origin_ts)
def alert_from(p, cut):
    raw = (p.reindex(P.index) > cut).fillna(False).values
    act = pd.Series(episodes_from_flags(raw, 12), index=P.index)
    return act.shift(1).fillna(False).reindex(m.idx).values.astype(bool)
rows = [metrics(sim.run(Recenter(2)), m, label='Recenter ±2 (no gate)', event_mask=em)]
for nm, al in [('Kumo', ka), ('Price abs(d)>=8', p8)]:
    r = metrics(sim.run(Gated(Recenter(2), al, 'freeze')), m, label=f'gate: {nm}', event_mask=em); r['alert_share'] = al.mean(); rows.append(r)
for nm, f in FC.items():
    if nm in ('ARIMA', 'Naive'): continue
    p = tailprob(f)
    lo, hi = 1e-6, 1.0
    for _ in range(40):  # bisection on cutoff to match the alert-hour share
        mid = (lo + hi) / 2; a = alert_from(p, mid)
        lo, hi = (mid, hi) if a.mean() > target else (lo, mid)
    a = alert_from(p, (lo + hi) / 2)
    r = metrics(sim.run(Gated(Recenter(2), a, 'freeze')), m, label=f'gate: {nm} P(abs d>=10, 6-12h)', event_mask=em)
    r['alert_share'] = a.mean(); r['episodes'] = len(episode_onsets(a)); rows.append(r)
# two-model gate: Kumo AND forecaster both high? -> union and intersection with our model
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.precision', 2)
print(T[['alert_share','episodes','net_vs_rebal_apr','hedged_sharpe','hedged_maxdd_bps','worst_day_bps','hedged_cvar5_daily_bps','event_hedged_bps','calm_hedged_apr']].to_string())
T.to_csv(OUT / 'table8_forecaster_gate.csv')
