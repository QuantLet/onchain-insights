<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: Onchain Insights

Published in: Onchain Insights : a study of Decentralized finance metrics for stablecoin depeg risk

Description: This study focuses on extensive open-access data available on the blockchain, and its use to predict depeg risks for various stablecoins. In particular, we focus on the dynamics of the liquidity curve from Uniswap's popular v3 liquidity pool protocol, focusing on the dominant stablecoin pool, USDC-USDT.


Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted: 25.01.2026

```

# Repo Instructions

Run this command to get up-to-date data :

```bash
bash update_data.sh
```

Install python requirements 

```bash
pip install -r requirements.txt
```


# Repo Structure and Example plots

## Quantlet 1: Binance USDE depeg

### Description and Output
This quantlet uses minute OHLC data from binance and block by block metrics on the USDE/USDT Uniswap liquidity pool to plot an animation higlighting the price divergence during the Ethena depeg event on October 10th 2025.

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/1.%20USDE%20binance%20depeg/ohlc.gif" alt="Image" />
</div>

### Recreate the plot

```bash
cd 1.\ USDE\ binance\ depeg/ ;
python plot_animation.py
```

## Quantlet 2: Curve liquidity pools
### Description and Output
This quantlet explores the theoretical invariant curve for Curvefi's Stableswap pools. It plots the different invariant curves dpeneding on the amplification parameter, as well as historical partial quotes and pool imbalance.
<div style="display: flex; width: 100%;">

<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/2.%20Curve%20liquidity%20pools/recent_past_3pool_uniswap.png" alt="Image" style="width: 50%; height: auto;" />

<img src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/2.%20Curve%20liquidity%20pools/stableswap_invariant_curves.png" alt="Image" style="width: 50%; height: auto;"/>
</div>

### Recreate the plots

Run the notebook :

```bash
2. Curve liquidity pools/code.ipynb
```


## Quantlet 3
### Description and Output
### Recreate the plot
## Quantlet 4
### Description and Output
### Recreate the plot
## Quantlet 5
### Description and Output
### Recreate the plot
## Quantlet 6
### Description and Output
### Recreate the plot
## Quantlet 7
### Description and Output
### Recreate the plot
## Quantlet 8
### Description and Output
### Recreate the plot
## Quantlet 9
### Description and Output
### Recreate the plot
## Quantlet 10
### Description and Output
### Recreate the plot