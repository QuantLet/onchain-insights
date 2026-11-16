"""Controls for the 'Kumo AND forecaster-ES' gate: price-filtered Kumo at matched alert share, random thinning of
Kumo alert hours at matched share (placebo), other forecasters' ES filters; bootstrap of daily differences."""
import numpy as np, pandas as pd
from data import make_market, event_mask, kumo_alert, OUT
from lpsim import Sim, metrics
from strategies import Recenter, Gated, episode_onsets
from run_all import load_forecasters, width_strategy, stationary_bootstrap_diff
from es_strategies import ESBand, es_interval, es_tail_distance
m = make_market(); sim = Sim(m); em = event_mask(m); ka = kumo_alert(m); FC = load_forecasters()
f = FC['Our model']; dist = es_tail_distance(f, m)
g_es = ka & (dist > 10); share = g_es.mean()
print(f'Kumo alone {ka.mean():.4f}, Kumo AND ourES>10 {share:.4f} ({len(episode_onsets(g_es))} spells)')
# price filter at matched share
absd = np.abs(m.depeg)
th = np.quantile(absd[ka], 1 - share / ka.mean())
g_px = ka & (absd >= th)
print(f'price filter |d|>={th:.2f} share {g_px.mean():.4f}')
cols = ['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps', 'hedged_cvar5_daily_bps', 'event_hedged_bps', 'calm_hedged_apr']
rows, daily = [], {}
def run(s, key):
    df = sim.run(s); r = metrics(df, m, label=key, event_mask=em); rows.append(r)
    daily[key] = df['hedged'].resample('1D').sum() / 1e4
    return r
W = lambda: width_strategy(f, m)
lo5, hi5, med6 = es_interval(f, m, 0.05, 6)
run(W(), 'Our width')
run(Gated(W(), ka, 'freeze'), 'Our width + Kumo')
real = run(Gated(W(), g_es, 'freeze'), 'Our width + Kumo∧ourES>10')
run(Gated(W(), g_px, 'freeze'), 'Our width + Kumo∧price (matched share)')
run(ESBand(lo5, hi5, med6, 0.5), 'Our ES band x0.5')
run(Gated(ESBand(lo5, hi5, med6, 0.5), g_es, 'freeze'), 'Our ES band x0.5 + Kumo∧ourES>10')
run(Recenter(2), 'Recenter ±2')
run(Gated(Recenter(2), ka, 'freeze'), 'Recenter ±2 + Kumo')
for nm in ['Our model', 'TimeXer', 'TiDE', 'GARCH', 'Rolling empirical']:
    g = ka & (es_tail_distance(FC[nm], m) > 10)
    r = run(Gated(Recenter(2), g, 'freeze'), f'Recenter ±2 + Kumo∧{nm} ES>10'); r['alert_share'] = g.mean()
R = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.precision', 2)
print(R[cols + (['alert_share'] if 'alert_share' in R else [])].to_string())
# placebo: random thinning of Kumo alert hours (contiguous spells kept intact) to the same share
rng = np.random.default_rng(11)
on = episode_onsets(ka); ends = np.flatnonzero(ka & ~np.r_[ka[1:], False])
spells = list(zip(on, ends))
P = []
for _ in range(300):
    a = np.zeros(m.n, bool); idx = rng.permutation(len(spells))
    for i in idx:
        s0, e0 = spells[i]
        if a.sum() + (e0 - s0 + 1) > share * m.n * 1.05: continue
        a[s0:e0 + 1] = True
        if a.sum() >= share * m.n: break
    P.append(metrics(sim.run(Gated(W(), a, 'freeze')), m, event_mask=em))
P = pd.DataFrame(P)
print('\nPlacebo: random subsets of Kumo episodes with the same alert-hour share (300 draws)')
for c in ['net_vs_rebal_apr', 'hedged_sharpe', 'hedged_sortino', 'hedged_maxdd_bps', 'worst_day_bps']:
    print(f'  {c:22s} real {real[c]:8.2f}  placebo median {P[c].median():8.2f}  [5%,95%] [{P[c].quantile(.05):.2f},{P[c].quantile(.95):.2f}]  p={(P[c] >= real[c]).mean():.3f}')
print('\nBootstrap of daily differences (bp/yr)')
for a_, b_ in [('Our width + Kumo∧ourES>10', 'Our width + Kumo'), ('Our width + Kumo∧ourES>10', 'Our width'),
               ('Our width + Kumo∧ourES>10', 'Our width + Kumo∧price (matched share)'), ('Our ES band x0.5', 'Our width'),
               ('Recenter ±2 + Kumo∧Our model ES>10', 'Recenter ±2 + Kumo')]:
    mu, lo, hi, p = stationary_bootstrap_diff(daily[a_] * 100, daily[b_] * 100, B=2000)
    print(f'  {a_} vs {b_}: {mu*365:+.1f} [{lo*365:+.1f}, {hi*365:+.1f}] p={p:.4f}')
R.to_csv(OUT / 'table11_es_gate_controls.csv'); P.to_csv(OUT / 'table11b_es_gate_placebo.csv')
