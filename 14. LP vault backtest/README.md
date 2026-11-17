```
Name of Quantlet: Onchain Insights - LP vault backtest

Published in: Onchain Insights - Onchain Monitoring of Stablecoin Depeg Risk

Description: Economic evaluation of the two-model framework on the Uniswap V3 USDC-USDT 0.01% pool. A
concentrated-liquidity vault (Gamma-style benchmark) is run hourly; the probabilistic forecaster sets its range
(the predictive 10-90% interval over the next 6 hours) and the early-warning classifier pauses repositioning during
its alert episodes. Net profit is measured against the rebalancing portfolio (fees + loss-versus-rebalancing -
costs). Controls: a 15 bp price trigger, a 24-hour oracle and random placebo schedules. Includes a 2024
out-of-sample check and a March 2023 USDC depeg stress scenario.

Keywords: Uniswap V3, liquidity provision, loss-versus-rebalancing, stablecoins, early warning,
probabilistic forecasting

Author: Owen Chaffard

Datafile: USDC_USDT_hourly_metrics.parquet, classifier operational_scores.parquet and cv_alert_outcomes.csv,
forecast artifacts preds_test_set.compact.npz
```

## Files

| File | Role |
|---|---|
| `config.py` | **All paths and parameters** (classifier scores and cutoff, forecast artifacts, windows, costs). Edit this when the models change. |
| `extract_quantiles.py` | Reduces each forecast artifact (origins x 24 x 256 quantiles) to the 26 levels used here, in `forecast_quantiles/`. |
| `data.py` | Pool panel, alert schedules (classifier, price trigger, oracle), 15 bp events. |
| `forecasts.py` | Quantile forecasts, predictive intervals, rolling empirical benchmark. |
| `lpsim.py` | Hourly vault simulator (fees, LVR, costs) and metrics. |
| `strategies.py` | Static range, Gamma-style recentring, forecaster range (asymmetric; symmetric variant), classifier pause. |
| `backtest.py` | Runs every backtest and writes `results/`. |
| `figures.py` | Writes the paper figures to `figures/` (PNG, transparent). |
| `paper_tables.py` | Writes LaTeX tables and number macros to `paper/generated/` and checks every qualitative claim made in the text. |
| `pq.py` | Pure-Python parquet reader, used only when pyarrow is not installed. |
| `paper/lp_backtest_insert.tex` | Text blocks for the manuscript, with their locations. |
| `paper/manuscript_lp_draft.tex` | Copy of `manuscript.tex` with the blocks inserted (for diffing). |

## Re-running with new model specifications

1. Point `config.py` to the new outputs: `KUMO_SCORES` and `KUMO_CUTOFF` (and `KUMO_CV_ALERTS` for the 2024 check),
   and `FORECAST_ARTIFACTS`. The holdout window is derived from the coverage of the classifier scores and the main
   forecaster.
2. Run, from this folder:
   ```bash
   python extract_quantiles.py      # once per new forecast artifact
   python backtest.py               # ~2 min; --quick uses 50 placebo schedules
   python figures.py
   python paper_tables.py           # prints CLAIM OK / CLAIM FAILS for every statement in the text
   ```
3. Copy `paper/generated/*.tex` into the folder `lp/` of the Overleaf project and `figures/lp_*.png` into its
   `figures/` folder. The text uses the macros in `lp/lp_numbers.tex`, so numbers update automatically; revise the
   text for any claim reported as failing.

## Accounting and timing

* Fee income per hour: `feesUSD x share of the hour's high-low range inside the position x L / (L_active + L)`.
* `LVR = V(P1) - V(P0) - x(P0)(P1 - P0) <= 0` (Milionis et al., 2022): loss relative to the rebalancing portfolio,
  which holds the position's USDC inventory and trades at market prices. Net profit = fees + LVR - costs, so the
  rebalancing portfolio has zero net profit by construction; `V_T - V_0 = sum(net) + sum(X_{t-1} dP_t)` holds exactly.
* Costs: $10 gas per withdrawal and deposit; the swap restoring the inventory of a new range pays the 1 bp pool fee
  plus 0.5 bp slippage.
* Decisions at the end of hour k for hour k+1. Price rules see P_k immediately; model outputs stamped k are used
  from the end of hour k+1 (one-hour data lag).
* The subgraph has no fee data before 2024-03-19 16:00 UTC: the 2024 check starts on 2024-03-20, and the March 2023
  stress scenario reports losses only.

## Outputs (`results/`)

`main.csv` (holdout comparison), `per_event.csv`, `regimes.csv` (fees/LVR/costs inside and outside event windows),
`placebo.csv`, `forecasters.csv` (every forecaster, asymmetric and symmetric ranges), `sensitivity.csv`,
`y2024.csv`, `stress.csv`, `meta.json` (windows, alert statistics), and `curves.npz` / `stress_curves.npz` for the
figures.
