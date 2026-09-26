<p align="center">
  <a href="https://quantlet.com">
    <img
      src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true"
      alt="Header Image"
      width="100%"
    />
  </a>
</p>

```
Name of Quantlet: Onchain Insights - Early Warning Classifier Cross Validation

Published in: Onchain Insights - Early Warning Classifier Cross Validation

Description: We train early warning models for depeg detection. Depeg is defined as an absolute deviation of the pool price above a given threshold. We apply 5-fold expanding window cross validation in order to compare the performance of common tree-based architectures. The dataset is comprised of features generated in previous quantlets describing onchain liqudiity conditions and broader market state. In particular we show the model performance for varying $\alpha$ tuning the Gegenbauer basis for the decomposition of the Uniswap USDC-USDT liquidity curve.

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted: 04.05.2026

Datafile: ./data/*
```

# Running the code

- Update data for all the repo. This shell script downloads the last release from the daily updated dataset and updates the corresponding files in all quantlets.

```bash
bash update_data.sh
```

- Move terminal into this specific quantlet:

```bash
cd 8.\ Early-Warning\ Model/
```

- Install requirements:

```bash
pip install -r requirements.txt
```

- The first shell script runs full 5-fold cross-validation for all models and automatically updates the preprocessed datasets:

```bash
bash compare_model_cv_15bp.sh
```

- This shell script also runs sensitivity analysis on the depeg threshold (defaults to 5 10 15 25 bps):

```bash
bash compare_model_cv.sh
```

- You can update the full retraining shell script with the best performing model in CV. The full retraining script then does last training plus the full suite of plots, including SHAP explanations:

```bash
bash full_retraining.sh
```

# Generated plots

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/plots_summary/threshold_15bps/heatmap_cv_event_utility.png"
    alt="Operational Utility Heatmap"
  />
</div>

<p align="center">
  <b>5-fold cross validation results</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/roc_pr.png"
    alt="Final retraining ROC and PR curves"
    width="100%"
  />
</div>

<p align="center">
  <b>Final retraining AUC/AUPRC</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/timeseries/predictions_over_time.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>24-hour ahead predicted depeg probability out-of-sample</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/event_level_recall_and_lead_time.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Event-level recall and alert lead time</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_realised_depeg_events.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Unique depeg events timeline</b>
</p>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_chronological_cv_splits.png"
    alt="Predictions over time"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_cv_events_by_fold.png"
    alt="Predictions over time"
    width="49%"
  />
</div>

<p align="center">
  <b>Cross Validation setup description</b>
</p>