```
Name of Quantlet: Onchain Insights - LP vault backtest

Published in: Onchain Insights - Onchain Monitoring of Stablecoin Depeg Risk

Description: Economic backtest of concentrated-liquidity (Gamma-style) LP vault strategies on the
Uniswap V3 USDC-USDT 0.01% pool driven by the two-model framework: the early-warning classifier
(Kumo, alpha=1, NP cutoff fixed before the holdout) gates the vault into a protective regime, and the
probabilistic forecaster sets the LP range width from its predictive quantiles. Strategies are compared
with standard benchmarks (static parity ranges, recentring on exit, volatility-scaled ranges), a
matched-budget price trigger, an oracle, random-alert placebos and alternative forecasters under an
identical decision rule.

Keywords: Uniswap V3, liquidity provision, loss-versus-rebalancing, stablecoins, early warning,
probabilistic forecasting, decision-based evaluation

Author: Owen Chaffard

Datafile: USDC_USDT_hourly_metrics.parquet, Kumo holdout operational_scores.parquet,
section-12 forecast artifacts (release section12-forecasts-v1)
```

## What is simulated

* **Pool / fees.** Hourly buckets of the 0.01% USDC-USDT pool (`4. Stablecoin liquidity ownership/USDC_USDT_hourly_metrics.parquet`).
  A position with liquidity *L* on ticks [a, b] earns `feesUSD x f_in_range x L / (L_active + L)`, where `f_in_range` is the
  overlap of the bucket's high-low range with [a, b].
* **PnL vs rebalancing.** Each hour: `hedged = fees + LVR - costs`, with `LVR = V(P1) - V(P0) - x(P0)(P1 - P0) <= 0`
  (Milionis et al., 2022, discretised hourly). This is the LP's PnL against a delta-matched rebalancing portfolio and
  removes the common USDC/USDT price exposure. The identity `V_T - V_0 = sum(hedged) + sum(x_{t-1} dP_t)` holds exactly.
* **Costs.** $10 gas per burn+mint, swap of the inventory imbalance at 1 bp pool fee + 0.5 bp slippage. Sensitivities:
  low/high cost, $100k / $5M vault, 50% of fees captured by JIT liquidity.
* **Timing (no look-ahead).** Decisions are taken at the end of bucket *k* for bucket *k+1*. Price rules react to
  the bucket close immediately; model signals stamped *k* are only used from the end of bucket *k+1* (1h data lag).
  Forecasts from origin *k-1* use horizons h >= 2.
* **Window.** 2025-01-15 16:00 to 2026-05-04 06:00 UTC (common coverage of classifier holdout and forecaster test set;
  5 eligible 15 bp events). 2024 (CV test folds, 18 events) is used for the classifier and for ex-ante benchmark choice.

## Strategies

| Family | Strategy | Rule |
|---|---|---|
| Standard | Static parity ±5 | fixed range [-5, +5] ticks around parity |
| Standard | Recenter ±w | Gamma-style: ±w ticks around the current tick, recentred when price exits |
| Standard | Vol-scaled recenter | half-width = 2 x trailing-24h depeg range / 2 |
| Classifier | + Kumo alert freeze | during alert episodes (12h quiet reset) do not rebalance |
| Classifier | + Kumo alert parity ±15 | on alert, move to a parity-anchored ±15 tick range until the episode ends |
| Control | + price |d|>=8 / oracle / placebo | same actions with a price trigger (matched alert budget), perfect foresight, or random alerts |
| Forecaster | forecast width | half-width from the q10-q90 interval over the next 6h, centred on the forecast median, recentred on exit or when the target width changes by >2x |
| Two-model | width + Kumo freeze / tail-range | forecaster sets the range; Kumo freezes it (or switches to the q1-q99 24h interval) during alerts |

## Running

```bash
# 1. once: reduce the section-12 forecast artifacts to the 26 quantile levels used here
python "14. LP vault backtest/extract_quantiles.py"
# 2. backtest, tables and figures -> 14. LP vault backtest/results/
python "14. LP vault backtest/run_all.py"          # add --quick for fewer placebo/bootstrap draws
```

The run uses `pandas.read_parquet` when pyarrow is installed and falls back to the bundled pure-Python reader (`pq.py`).
Our model (`model_logs/final_retrain_run/preds_test_set.compact.npz`) is picked up automatically when its quantiles
are extracted; otherwise TiDE is used as the main forecaster.

## Outputs

`run_all.py`: `table1_main` (strategy comparison), `table2_placebo` (random-alert placebo), `table3_forecasters` and `table3b_bootstrap`
(forecaster horse race under one decision rule, stationary bootstrap of daily PnL differences), `table4_2024`
(classifier in the 2024 CV period), `table5_sensitivity`, and figures `fig1_cumulative_pnl`, `fig2_event_zoom`,
`fig3_risk_return` (PNG + PDF).

Additional analyses (run after `run_all.py`; same environment variables):

* `frontier.py` - matched-width comparison: each forecaster's q10-q90 half-width scaled by k in 0.25-2.0, net PnL
  interpolated at common average widths, against constant-width recentring (`table6_width_frontier`,
  `table6b_matched_width`, `fig4_width_frontier`).
* `tail_marginal.py` - fixed-horizon tail decision: P(|d| >= 10 or 15 bp at h = 6, 12, 24) from each forecaster;
  Brier, log score, AUC and relative economic value of a cost-loss protection rule (`table7b_tail_marginal`).
* `tail_value.py` - same for a breach anywhere in the next 24h, using the maximum of the marginals (a lower bound on
  the path probability, which penalises calibrated marginals; `table7_tail_value`).
* `fc_gate.py` - each forecaster's tail probability used as the vault gate at Kumo's alert-hour budget
  (`table8_forecaster_gate`).
* `compounding.py` - main comparison with fees held as cash (base), reinvested at each repositioning, or also
  collected and reinvested whenever accrued fees exceed 0.1% of capital (`table9_compounding`).

`paper/economic_evaluation.tex` holds the text, tables and figure blocks for the manuscript (with insertion points),
`paper/economic_evaluation_refs.bib` the four new references, and `paper/figures/` the figure PDFs.

Tail-based (expected-shortfall) strategies:

* `es_strategies.py` - ES from the quantile forecasts; ES band (asymmetric range from the 5% ES over 6h), ES-sized capital,
  classifier gate filtered by the 24h ES distance from the current price, ES dislocation range on alert.
* `run_es.py` - all ES strategies for every forecaster (`table10_es_strategies`).
* `run_es_gate.py`, `es_bootstrap.py` - controls for the ES-filtered gate: price filter at matched alert share, random
  thinning of classifier episodes (placebo), stationary bootstrap (`table11_*`).
* `es_combined_all.py` - ES band + ES-filtered gate for every forecaster with bootstrap (`table12_es_combined_all`).

Metrics now include the Sortino ratio of hourly net PnL (`hedged_sortino`; daily version `hedged_sortino_daily`).
