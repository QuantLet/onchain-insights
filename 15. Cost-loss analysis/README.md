```
Name of Quantlet: Onchain Insights - Cost-loss analysis

Published in: Onchain Insights - Onchain Monitoring of Stablecoin Depeg Risk

Description: Economic value of acting on the early-warning classifier and the probabilistic forecaster in the
cost-loss framework of forecast verification (Murphy 1977; Richardson 2000), without a trading simulation. Each
warning costs c and each unanticipated 15 bp depeg costs L; relative economic value is reported for every
cost-loss ratio c/L, for all candidate classifiers in cross-validation, the selected policy over the holdout, a
price-precursor benchmark, the forecast exceedance probabilities of all forecasters, and a joint rule in which
the forecaster screens the classifier's warnings.

Keywords: cost-loss ratio, relative economic value, early warning, probabilistic forecasting, stablecoins

Author: Owen Chaffard

Datafile: classifier CV outputs (five_alert_budget_model_comparison.csv, five_alert_budget_fold_results.csv),
holdout operational_scores.parquet, forecast quantiles and pool panel (via folder 14)
```

## Files

| File | Role |
|---|---|
| `cost_loss.py` | Computes everything and writes `results/`. Reuses the data, alert and forecast modules of `14. LP vault backtest` (paths in its `config.py`). |
| `figures.py` | `figures/cl_classifier_value.png`, `cl_forecaster_value.png`, `cl_joint_value.png` (PNG, transparent). |
| `paper_tables.py` | LaTeX tables and number macros in `paper/generated/`; prints CLAIM OK / CLAIM FAILS for every statement in the text. |
| `paper/cost_loss_insert.tex` | Text blocks for the manuscript, with their locations. |
| `paper/cost_loss_refs.bib` | Murphy (1977) and Richardson (2000). |

## Running

```bash
python cost_loss.py
python figures.py
python paper_tables.py
```
Then copy `paper/generated/*.tex` to the folder `cl/` of the Overleaf project, `figures/cl_*.png` to its `figures/`
folder, and add `\input{cl/cl_numbers}` to the preamble.

## Definitions

* **Warnings.** An alert episode is a warning and triggers one action of cost c. An eligible 15 bp event is
  anticipated if an alert episode begins 2-24 h before its first breach (the paper's timely recall); otherwise it
  costs L. With D anticipated events out of N and W warnings, `V(c/L) = (D - (c/L) W) / (N (1 - c/L))`, positive iff
  c/L < D/W. Cross-validation counts come from the fold results of the selected run (all alert episodes: false,
  timely and right-censored); holdout counts from the operational scores.
* **Price precursor.** Warn when |d| >= 8, 10 or 12 bp (12 h quiet reset), evaluated on the same windows.
* **Forecasts.** Protect against |d_{t+h}| >= theta when the forecast exceedance probability exceeds c/L; value
  against the better of always/never protecting (Richardson 2000).
* **Joint.** Act on a classifier warning only if the forecast probability of a 15 bp breach 2-24 h ahead (largest
  marginal exceedance probability over those horizons) exceeds c/L. Holdout only.
