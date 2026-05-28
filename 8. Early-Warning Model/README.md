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
bash compare_model_cv.sh
```

- You can update the full retraining shell script with the best performing model in CV. The full retraining script then does last training plus the full suite of plots, including SHAP explanations:

```bash
bash full_retraining.sh
```

# Generated plots

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25/plots_summary/heatmap_cv_auc.png"
    alt="CV AUC Heatmap"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25/plots_summary/heatmap_cv_auprc.png"
    alt="CV AUPRC Heatmap"
    width="49%"
  />
</p>

<p align="center">
  <b>5-fold cross validation results</b>
</p>

<br>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/roc_pr.png"
    alt="Final retraining ROC and PR curves"
    width="100%"
  />
</p>

<p align="center">
  <b>Final retraining AUC/AUPRC</b>
</p>

<br>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/timeseries/predictions_over_time.png"
    alt="Predictions over time"
    width="100%"
  />
</p>

<p align="center">
  <b>24-hour ahead predicted depeg probability out-of-sample</b>
</p>

<br>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/shap/shap_beeswarm_global.png"
    alt="SHAP beeswarm summary plot"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/feature_importance/feature_importance_native.png"
    alt="Native model feature importance"
    width="49%"
  />
</p>

<p align="center">
  <b>SHAP beeswarm summary plot / Native model feature importance</b>
</p>

<br>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/shap/shap_scatter_curve_entropy.png"
    alt="SHAP scatter plot for curve entropy"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/shap/shap_scatter_tangent_up.png"
    alt="SHAP scatter plot for tangent up"
    width="49%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/shap/shap_scatter_tvlUSD_500.png"
    alt="SHAP scatter plot for tvl USD 500"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison_2026-05-25_full_retraining/random_forest_alpha_0.1_fullfeatures_top2_cv_auc/artifacts/plots/shap/shap_swap_count_100.png"
    alt="SHAP scatter plot for swap count 100"
    width="49%"
  />
</p>

<p align="center">
  <b>SHAP correlation scatter plots</b>
</p>