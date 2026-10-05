<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: Onchain Insights - Curve liquidity pools

Published in: Onchain Insights - Curve liquidity pools

Description: This Quantlet computes a suite of probabilistic diagnostics in order to evaluate the probabilistic forecasting abilities of our model for depeg in the Uniswap pool.

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted:  06.05.2026

Datafile: curve_3pool_hourly.parquet, hourly_pool_state_full.parquet

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

The tail diagnostics plots can be ran from the following notebook (after downloading the forecast artifacts):

```bash
./tail_diagnostics.ipynb
```

Run this for reproducing the benchmark tables in the paper:

```bash
python ./tables/tables_full_benchmark.py
python ./tables/tables_DM_tests.py
```
## Outputs and Figures

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/12. Evaluation of Probabilistic Forecasts/var_exceedance_counts.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Value at Risk exceedance count plot</b>
</p>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/12. Evaluation of Probabilistic Forecasts/PIT_VAR_ES_calibration.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Tail calibration diagnostic plots</b>
</p>

| Model | CRPS | twCRPS | twCRPS lower (<-10) | twCRPS upper (>10) | QL 1% | QL 5% | QL 95% | QL 99% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAINT | **0.8205** | **0.0665** | **0.0121** | **0.0545** | **0.0706** | **0.2081** | **0.1952** | **0.0611** |
| GARCH | 2.2140 | 0.1060 | 0.0306 | 0.0754 | 0.1123 | 0.3884 | 0.3229 | 0.1057 |
| TimeXer | 0.8787 | 0.0883 | 0.0206 | 0.0677 | 0.0811 | 0.2225 | 0.2062 | 0.0685 |
| TiDE | 0.9702 | 0.0775 | 0.0132 | 0.0643 | 0.1560 | 0.3061 | 0.3009 | 0.1555 |
| ARIMA | 4.4917 | 1.3344 | 0.6747 | 0.6597 | 0.4212 | 1.4877 | 1.4975 | 0.4232 |
| Naive | 5.4997 | 2.1775 | 1.1169 | 1.0607 | 0.5190 | 1.8347 | 1.8347 | 0.5190 |

<p align="center">
  <b>Full benchmark table</b>
</p>

| Metric   | Benchmark   |   Mean loss proposed |   Mean loss benchmark |   Improvement (%) |   Mean diff proposed-minus-benchmark |   DM p-value proposed better |   Bootstrap CI low |   Bootstrap CI high |
|:---------|:------------|---------------------:|----------------------:|------------------:|-------------------------------------:|-----------------------------:|-------------------:|--------------------:|
| CRPS     | GARCH       |               1.1739 |                2.2943 |           48.8337 |                              -1.1204 |                       0      |            -1.3066 |             -0.9403 |
| CRPS     | TimeXer     |               1.1739 |                1.2535 |            6.3467 |                              -0.0796 |                       0      |            -0.1117 |             -0.049  |
| CRPS     | TiDE        |               1.1739 |                1.38   |           14.9357 |                              -0.2061 |                       0      |            -0.2386 |             -0.1751 |
| CRPS     | ARIMA       |               1.1739 |                5.8447 |           79.9145 |                              -4.6707 |                       0      |            -4.7678 |             -4.564  |
| CRPS     | Naive       |               1.1739 |                8.0142 |           85.352  |                              -6.8403 |                       0      |            -6.9451 |             -6.7275 |
| twCRPS   | GARCH       |               0.0903 |                0.1117 |           19.097  |                              -0.0213 |                       0.0047 |            -0.0399 |             -0.0062 |
| twCRPS   | TimeXer     |               0.0903 |                0.1286 |           29.7649 |                              -0.0383 |                       0      |            -0.0575 |             -0.0224 |
| twCRPS   | TiDE        |               0.0903 |                0.1049 |           13.9221 |                              -0.0146 |                       0.0014 |            -0.0261 |             -0.0045 |
| twCRPS   | ARIMA       |               0.0903 |                2.096  |           95.69   |                              -2.0056 |                       0      |            -2.0275 |             -1.9783 |
| twCRPS   | Naive       |               0.0903 |                3.9915 |           97.7368 |                              -3.9011 |                       0      |            -3.952  |             -3.8481 |

<p align="center">
  <b>Pairwise model vs benchmark comparisons, Diebold Mariano tests and block bootstrap results.</b>
</p>