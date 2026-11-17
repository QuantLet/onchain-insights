"""Reactions to an alert: pause, exit to 50/50, widen to the ES band, peg-reversion trade (buy the cheap token).

    python reactions.py   -> results/reactions.csv, results/reactions_stress.csv
"""
import numpy as np, pandas as pd
import config as C
from data import market, kumo_alert, price_alert, events
from forecasts import load_forecasters, es_band, median_at, QForecast
from lpsim import metrics
from strategies import Recenter, ForecastInterval, AlertReaction
from backtest import holdout_window, sim

REACTIONS = ['freeze', 'exit', 'widen', 'widen-hold', 'tiered', 'revert', 'widen+revert']


def reaction_set(m, f, alert, tag):
    lo, hi = f.interval(m); el, eh = es_band(f, m); med = median_at(f, m, 24)
    F = lambda: ForecastInterval(lo, hi)
    V = {f'Forecaster | {tag} | none': F}
    for r in REACTIONS:
        V[f'Forecaster | {tag} | {r}'] = lambda r=r: AlertReaction(F(), alert, r, el, eh, med)
    V[f'Forecaster | {tag} | revert (no forecast filter)'] = lambda: AlertReaction(F(), alert, 'revert', el, eh, None)
    return V


def main():
    m = market(*holdout_window()); s = sim(m); ev = events(m); f = load_forecasters(m)[C.MAIN_FORECASTER]
    ka, p15 = kumo_alert(m), price_alert(m)
    V = {}
    V.update(reaction_set(m, f, ka, 'classifier'))
    V.update(reaction_set(m, f, p15, 'price 15bp'))
    always = np.ones(m.n, bool)
    lo, hi = f.interval(m); el, eh = es_band(f, m)
    V['Forecaster | always | revert at 8 bp (peg-arbitrage rule)'] = lambda: AlertReaction(ForecastInterval(lo, hi), always, 'revert', el, eh, None, entry=8)
    V['Forecaster | always | revert at 15 bp (peg-arbitrage rule)'] = lambda: AlertReaction(ForecastInterval(lo, hi), always, 'revert', el, eh, None, entry=15)
    V['Gamma | none | none'] = lambda: Recenter(C.GAMMA_HALF_WIDTH)
    rows = []
    for k, mk in V.items():
        r = metrics(s.run(mk()), m, ev, C.EVENT_WINDOW, label=k); rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(C.OUT / 'reactions.csv')
    pd.set_option('display.width', 260)
    print(T[['net', 'inventory', 'vs_hodl', 'sharpe', 'vs_hodl_sharpe', 'event_dd', 'vs_hodl_event_dd', 'max_dd', 'vs_hodl_max_dd',
             'vs_hodl_worst_day', 'ev_fees', 'ev_lvr', 'time_out_of_lp']].round(2).to_string())
    # stress scenario (no classifier, no fee data): price-gated reactions, total value vs HODL
    ms = market(*C.STRESS_WINDOW); s3 = sim(ms); ev3 = events(ms)
    fr = QForecast.rolling_empirical(ms.idx[0] - pd.Timedelta(hours=2), ms.idx[-1] + pd.Timedelta(hours=1))
    p3 = price_alert(ms); rows = []
    for k, mk in reaction_set(ms, fr, p3, 'price 15bp').items():
        r = metrics(s3.run(mk()), ms, ev3, C.EVENT_WINDOW, label=k.replace('Forecaster', 'Forecaster (rolling empirical)')); rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(C.OUT / 'reactions_stress.csv')
    print('\nMarch 2023 (losses only: no fee data), bp of capital over the month')
    print((S[['net', 'inventory', 'vs_hodl', 'vs_hodl_max_dd', 'vs_hodl_worst_day']] * np.array([ms.years, ms.years, ms.years, 1, 1])).round(1).to_string())


if __name__ == '__main__':
    main()
