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
Name of Quantlet: Onchain Insights

Published in: Onchain Insights : a study of Decentralized finance metrics for stablecoin depeg risk

Description: This study focuses on extensive open-access data available on the blockchain, and its use to predict depeg risks for various stablecoins. In particular, we focus on the dynamics of the liquidity curve from Uniswap's popular v3 liquidity pool protocol, focusing on the dominant stablecoin pool, USDC-USDT.

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted: 25.01.2026
```
# Table of Contents

| Quantlet | Dedicated README |
|---|---|
| [Quantlet 1: Binance USDE depeg](#quantlet-1-binance-usde-depeg) | [View Quantlet](./1.%20USDE%20binance%20depeg/README.md) |
| [Quantlet 2: Curve liquidity pools](#quantlet-2-curve-liquidity-pools) | [View Quantlet](./2.%20Curve%20liquidity%20pools/README.md) |
| [Quantlet 3: Uniswap liquidity curve](#quantlet-3-uniswap-liquidity-curve) | [View Quantlet](./3.%20Uniswap%20liquidity%20curve/README.md) |
| [Quantlet 4: Stablecoin Liquidity Ownership](#quantlet-4-stablecoin-liquidity-ownership) | [View Quantlet](./4.%20Stablecoin%20liquidity%20ownership/README.md) |
| [Quantlet 5: Functional PCA analysis of the liquidity curve](#quantlet-5-functional-pca-analysis-of-the-liquidity-curve) | [View Quantlet](./5.%20Functional%20PCA%20analysis%20of%20the%20liquidity%20curve/README.md) |
| [Quantlet 6: Legendre basis Decomposition](#quantlet-6-legendre-basis-decomposition) | [View Quantlet](./6.%20Legendre%20basis%20decomposition/README.md) |
| [Quantlet 7: Gegenbauer Polynomials](#quantlet-7-gegenbauer-polynomials) | [View Quantlet](./7.%20Gegenbauer%20Polynomials/README.md) |
| [Quantlet 8: Early-Warning Model](#quantlet-8-early-warning-model) | [View Quantlet](./8.%20Early-Warning%20Model/README.md) |
| [Quantlet 9: Parametric Quantile Function Characterisation](#quantlet-9-parametric-quantile-function-characterisation) | [View Quantlet](./9.%20Parametric%20quantile%20function%20characterisation/README.md) |
| [Quantlet 10: Forecasting architecture](#quantlet-10-forecasting-architecture) | [View Quantlet](./10.%20Forecasting%20architecture/README.md) |

# Repo Instructions

Run this command to get up-to-date data:

```bash
bash update_data.sh
```

Install python requirements:

```bash
pip install -r requirements.txt
```

# Repo Structure and Example plots

## Quantlet 1: Binance USDE depeg

### Description and Output

This quantlet uses minute OHLC data from binance and block by block metrics on the USDE/USDT Uniswap liquidity pool to plot an animation higlighting the price divergence during the Ethena depeg event on October 10th 2025.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/1.%20USDE%20binance%20depeg/ohlc.gif"
    alt="Binance USDE depeg animation"
    width="100%"
  />
</p>

### Recreate the plot

```bash
cd 1.\ USDE\ binance\ depeg/ ;
python plot_animation.py
```

## Quantlet 2: Curve liquidity pools

### Description and Output

This quantlet explores the theoretical invariant curve for Curvefi's Stableswap pools. It plots the different invariant curves dpeneding on the amplification parameter, as well as historical partial quotes and pool imbalance.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/2.%20Curve%20liquidity%20pools/recent_past_3pool_uniswap.png"
    alt="Recent past 3pool Uniswap"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/2.%20Curve%20liquidity%20pools/stableswap_invariant_curves.png"
    alt="Stableswap invariant curves"
    width="49%"
  />
</p>

### Recreate the plots

Run the notebook:

```bash
2. Curve liquidity pools/code.ipynb
```

## Quantlet 3: Uniswap liquidity curve

### Description and Output

This quantlet provides plots for viewing the time-varying liquidity distribution in the Uniswap v3 USDC-USDT pool. The liquidity curve is reconstructed from the Uniswap subgraph and we highlight its dynamics, as well as swap size impact curve, whicih are relevant to liqudiity risk for stablecoin depeg.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/3.%20Uniswap%20liquidity%20curve/liq_curve.gif"
    alt="Uniswap liquidity curve animation"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/3.%20Uniswap%20liquidity%20curve/liquidity_cliff.png"
    alt="Liquidity cliff"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/3.%20Uniswap%20liquidity%20curve/swap_size_impact.png"
    alt="Swap size impact"
    width="49%"
  />
</p>

### Recreate the plot

Run the notebook:

```bash
3.\ Uniswap\ liquidity\ curve/code.ipynb
```

## Quantlet 4: Stablecoin Liquidity Ownership

### Description and Output

This quantlet analyses ownership graphs of stablecoin liquidity on the 3pool and Uniswap. We first higlight the gap in TVL between the two venues, which is explained in part by Curve's liquidity being 70 owned by a hacker's cold wallet. We further highlight the distribution of whales on Curve's main stablecoin pool and the importance of metapools for the marketcap of the 3CRV LP token. Furthermore, we analyse the rise of MEV activity on the USDC-USDT v3 pool, showing a consistent volume of JIT (Just In Time) liquidity starting sometime in 2024. JIT liquidity is detected through block matching of mint and burn orders and we show it account for a major part of non-NFPM order flow.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/4.%20Stablecoin%20liquidity%20ownership/3pool_uniswap_volume_tvl.png"
    alt="3pool Uniswap volume TVL"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/4.%20Stablecoin%20liquidity%20ownership/3CRV_liquidity_distribution.png"
    alt="3CRV liquidity distribution"
    width="49%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/4.%20Stablecoin%20liquidity%20ownership/liquidity_curve_24883875.png"
    alt="Liquidity curve"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/4.%20Stablecoin%20liquidity%20ownership/nfpm_in_range_share_over_time.png"
    alt="NFPM in-range share over time"
    width="49%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/4.%20Stablecoin%20liquidity%20ownership/jit_share_over_time.png"
    alt="JIT share over time"
    width="50%"
  />
</p>

### Recreate the plots

Run the notebook:

```bash
4.\ Stablecoin\ liquidity\ ownership/code.ipynb
```

## Quantlet 5: Functional PCA analysis of the liquidity curve

### Description and Output

This quantlet contains plots related to the functional PCA analysis of the log-liquidity curve. FPCA is applied in a rolling window manner to study stability of the PCs and reliability of the decomposition through time using cumulative partial variance explained.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/5.%20Functional%20PCA%20analysis%20of%20the%20liquidity%20curve/uniswap_liquidity_surface_comparison.png"
    alt="Uniswap liquidity surface comparison"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/5.%20Functional%20PCA%20analysis%20of%20the%20liquidity%20curve/4PCs_pegcentered.png"
    alt="Four principal components peg centered"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/5.%20Functional%20PCA%20analysis%20of%20the%20liquidity%20curve/4PCs_pricecentered.png"
    alt="Four principal components price centered"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/5.%20Functional%20PCA%20analysis%20of%20the%20liquidity%20curve/uniswap_liquidity_pca_analysis.png"
    alt="Uniswap liquidity PCA analysis"
    width="100%"
  />
</p>

### Recreate the plots

Run the notebook:

```bash
5. Functional PCA analysis of the liquidity curve/code.ipynb
```

## Quantlet 6: Legendre basis Decomposition

### Description and Output

This quantlet creates plots evaluating the decomposition of the log-Liquidity curve into an interpretable Legendre polynomial basis. We specifically compare the interpretability of a peg and price centered decomposition. Showing that while a price centered decomposition might look more stable it is unfit for stablecoin study due to the static mass of liquidity at and around peg.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/6.%20Legendre%20basis%20decomposition/legendre_reconstruction_pegvsprice.png"
    alt="Legendre reconstruction peg versus price"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/6.%20Legendre%20basis%20decomposition/legendre_scores_pegvsprice.png"
    alt="Legendre scores peg versus price"
    width="100%"
  />
</p>

### Recreate the plot

Run the notebook:

```bash
6. Legendre basis decomposition/code.ipynb
```

## Quantlet 7: Gegenbauer Polynomials

### Description and Output

This quantlet documents the extension of the Legendre decomposition into more general Gegenbauer polynomial bases. The effect of the $\alpha$ parameter, allowing focus to be put on different parts of the curve, is evaluated and highlighted through shock effects of the log-liqudiity profile. The $\alpha$ parameter will be used as a hyperparameter for subsequent model allowing them to tune tail focus of the decomposition due to its control over the root density of the basis components.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/7.%20Gegenbauer%20Polynomials/gegenbauer_roots_weights.png"
    alt="Gegenbauer roots and weights"
    width="100%"
  />
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/7.%20Gegenbauer%20Polynomials/gegenbauer_scores_comparison.png"
    alt="Gegenbauer scores comparison"
    width="100%"
  />
</p>

### Recreate the plots

Run the notebook:

```bash
7. Gegenbauer Polynomials/code.ipynb
```

## Quantlet 8: Early-Warning Model

### Description and Output

This quantlet provides code to run full cross-validation analysis of common tree-based architectures on the previously built dataset of stablecoin liquidity data. The binary task is that of predicting a depeg deviation above 15 bps in the next 24 hours. The scripts allow Cross-validation for model selection and final retraining including a full suite of diagnostics for the final model's performance and explanations through internal feature impartance and SHAP explanations.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/roc_pr.png"
    alt="Final retraining ROC and PR curves"
    width="100%"
  />
</p>

<p align="center">
  <b>Final retraining AUC/AUPRC</b>
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_curve_entropy.png"
    alt="SHAP scatter plot curve entropy"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8.%20Early-Warning%20Model/lightning_logs/default/random_forest_alpha_0.1_fullfeatures/artifacts/plots/shap/shap_scatter_tangent_up.png"
    alt="SHAP scatter plot tangent up"
    width="49%"
  />
</p>

<p align="center">
  <b>SHAP correlation scatter plots</b>
</p>

### Recreate the plot

View the detailed instructions in the README:

```bash
8. Early-Warning Model/README.md
```

## Quantlet 9: Parametric Quantile Function Characterisation

### Description and Output

In this quantlet we evaluate the distribution of stablecoin depeg basis points. ARIMA/GARCH residuals show poor fit against Gaussian/Student's t innovations. In order to circumvent this issue we train a Neural Network to regress the quantile function in a non-distributional manner.
We show the use of both Chebyshev and I-spline bases for quantile function regression, and their advantages/drawbacks. We also introduce the threshold weighted CRPS used to favor rare extreme events in the diustributional calibration. 
Lastly we showcase the addition of spliced GPD tails for modeling extreme events/ closed form VaR+ES estimation.

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9.%20Parametric%20quantile%20function%20characterisation/ispline_basis_power_tails.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9.%20Parametric%20quantile%20function%20characterisation/ispline_basis_uniform.png" alt="Image" />
</div>
<p align="center">
  <b>I spline bases generated with uniform and power-tails knots</b>
</p>
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9.%20Parametric%20quantile%20function%20characterisation/quantile_function_comparison.png" alt="Image" />
</div>
<p align="center">
  <b>Quantile function tail focus based on knot density</b>
</p>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9.%20Parametric%20quantile%20function%20characterisation/gpd_spliced_quantile_logit.png" alt="Image" />
</div>
<p align="center">
  <b>Spliced Quantile function with parametric GPD tails</b>
</p>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9.%20Parametric%20quantile%20function%20characterisation/chaining_function.png" alt="Image" />
</div>

<p align="center">
  <b>Weight and Chaining function of the threshold weighted CRPS</b>
</p>


### Recreate the plot

Run the notebook:

```bash
9. Parametric quantile function characterisation/code.ipynb
```

## Quantlet 10: Forecasting architecture

### Description and Output

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/10.%20Forecasting%20architecture/architecture_ML.png" alt="Image" />
</div>
<p align="center">
  <b>Our custom Neural Network Forecasting architecture</b>
</p>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/10.%20Forecasting%20architecture/hard_concrete_plot.png" alt="Image" />
</div>

<p align="center">
  <b>L0 regularisation with hard concrete gating</b>
</p>
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/10.%20Forecasting%20architecture/nhits_stack_decomposition_fixed.png" alt="Image" />
</div>
<p align="center">
  <b>NHITS encoder stack decomposition</b>
</p>

### Recreate the plot

Run the notebook:

```bash
10. Forecasting architecture/code.ipynb
```