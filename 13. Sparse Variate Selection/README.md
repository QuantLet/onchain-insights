<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: Onchain Insights - Sparse Variate Selection

Published in: Onchain Insights - Sparse Variate Selection

Description: This Quantlet examines the sparse covariate selection in our forecasting model. The variate selection architecture is a context-aware Gated Residual Network (introduced by Lim et al. in https://arxiv.org/abs/1912.09363) which uses sparsemax activation and entropy regularisation to enforce a sparse representation. 

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted:  06.10.2026

```

## Large forecast artifacts

This repo contains large forecast artifacts, stored in a release for ease of reproducibility.In order to populate a clone, download and extract all the release packages; their stored paths put each file back in the expected folder:

```bash
section='12. Evaluation of Probabilistic Forecasts'
downloads="$(mktemp -d)"
gh release download section12-forecasts-v1 \
  --repo QuantLet/onchain-insights \
  --pattern 'section12-*.tar.gz' \
  --dir "$downloads"
for asset in "$downloads"/section12-*.tar.gz; do
  tar -xzf "$asset" -C "$section"
done
```

This requires the GitHub CLI (`gh`) authenticated to an account with access to
the repository.

## Running the code 

The sparse selection diagnostics plots can be ran from the following notebook (after downloading the forecast artifacts):

```bash
./code.ipynb
```

## Outputs and Figures

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/13. Sparse Variate Selection/fig1_variate_support_weight.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Variate Support Weight</b>
</p>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/13. Sparse Variate Selection/fig3_selection_raster.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Raster plot of variate selection at test origins</b>
</p>
