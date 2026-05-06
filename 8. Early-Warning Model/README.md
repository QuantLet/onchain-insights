<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

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

- Update data for all the repo (this shell script downloads the last release from the daily updated dataset and updates the corresponding files in all quantlets)

```bash
bash update_data.sh
```

- move terminal in this specific quantlet :

```bash
cd 8.\ Early-Warning\ Model/
```

- Install requirements:

```bash
pip install -r requirements.txt
```

- The first shell script runs full 5-fold cross-validation for all models (automatically updates the preprocessed datasets):

```bash
bash compare_model_cv.sh
```

- You can update the full retraining shell script with the best performing model in CV, the full retraining script then does last training + full suite of plots including shap explanations:

```bash
bash full_retraining.sh
```

# Generated plots

<div style="display: flex; width: 100%;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison/plots_summary/heatmap_cv_auc.png"
    alt="CV AUC Heatmap"
    style="width: 50%; height: auto;"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/cv_model_comparison/plots_summary/heatmap_cv_auprc.png"
    alt="CV AUPRC Heatmap"
    style="width: 50%; height: auto;"
  />
</div>
<center> <b> 5-fold cross validation results </b></center>


<div style="display: flex; width: 100%;margin-top: 10vh;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/roc_pr.png"
  />
</div>

<center> <b> Final retraining AUC/AUPRC </b></center>

<div style="display: flex; width: 100%;margin-top: 10vh;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/timeseries/predictions_over_time.png"
  />
</div>

<center> <b> 24-hour ahead predicted depeg probability out-of-sample </b></center>

<div style="display: flex; width: 100%;margin-top: 10vh;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_beeswarm_global.png"
    style="width: 50%; height: auto;"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/feature_importance/feature_importance_native.png"
    style="width: 50%; height: auto;"
  />
</div>

<center> <b> SHAP beeswarm summary plot / Native model feature importance </b></center>

<div style="display: flex; width: 100%;margin-top: 10vh;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_curve_entropy.png"
    style="width: 50%; height: auto;"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_tangent_up.png"
    style="width: 50%; height: auto;"
  />
</div>
<div style="display: flex; width: 100%;margin-top: 0vh;">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_tvlUSD_500.png"
    style="width: 50%; height: auto;"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_swap_count_100.png"
    style="width: 50%; height: auto;"
  />
</div>

<center> <b> SHAP beeswarm summary plots </b></center>
