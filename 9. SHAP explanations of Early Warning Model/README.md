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

Name of Quantlet: Onchain Insights - SHAP explanations of Early Warning Model

Published in: Onchain Insights - SHAP explanations of Early Warning Model

Description: The best performing Early Warning model is selected from the previous quantlet. SHAP explanations are produced to explain its outputs, including global summaries and local waterfalls. SHAP importance of Economic groups is highlighted.

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted:  25.09.2026
```

# Running the code

Run the notebook: 

```bash 
./SHAP_details.ipynb
```

# Generated Plots

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9. SHAP explanations of Early Warning Model/shap_beeswarm.png"
    alt="SHAP beeswarm"
  />
</div>

<p align="center">
  <b>Summary plot of SHAP feature attribution</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9. SHAP explanations of Early Warning Model/shap_beeswarm_net_positive.png"
    alt="SHAP beeswarm"
  />
</div>

<p align="center">
  <b>Summary plot of SHAP feature attribution to depeg alerts</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9. SHAP explanations of Early Warning Model/shap_grouped_importance_sum_abs.png"
    alt="SHAP beeswarm"
  />
</div>

<p align="center">
  <b>SHAP attribution by economic feature group</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9. SHAP explanations of Early Warning Model/shap_scatter_plots_liq.png"
    alt="SHAP beeswarm"
  />
</div>

<p align="center">
  <b>SHAP scatter plots for selected liquidity features</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/9. SHAP explanations of Early Warning Model/shap_scatter_plots_volatility.png"
    alt="SHAP beeswarm"
  />
</div>

<p align="center">
<b>SHAP scatter plots for selected liquidity curve features</b>
</p>

<br>