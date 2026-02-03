# download and extract new data from latest release
gh release download --repo MSCA-DN-Digital-Finance/stablecoin-onchain-data --pattern "*" --dir ./ --clobber;
for type in aave curve eth_blocks uniswap

do 
unzip -o $type\_data.zip;
rm $type\_data.zip;

done;

# copy input files to quantlets
cp -rf ./data/Curve/curve_3pool_hourly.parquet ./2.\ Curve\ liquidity\ pools;
cp -rf ./data/Uniswap/hourly_pool_state_full.parquet ./2.\ Curve\ liquidity\ pools;

cp -rf ./data/Uniswap/hourly_pool_state_full.parquet ./3.\ Uniswap\ liquidity\ curve;
cp -rf ./data/Uniswap/hourly_liquidity_full.parquet ./3.\ Uniswap\ liquidity\ curve;
cp -rf ./data/Uniswap/hourly_liquidity_pricecentered_full.parquet ./3.\ Uniswap\ liquidity\ curve;

cp -rf ./data/Curve/curve_3pool_hourly.parquet ./4.\ Stablecoin\ liquidity\ concentration;
cp -rf ./data/Uniswap/USDC_USDT_hourly_metrics.parquet ./4.\ Stablecoin\ liquidity\ concentration;

cp -rf ./data/Uniswap/hourly_pool_state_full.parquet ./5.\ Functional\ PCA\ analysis\ of\ the\ liquidity\ curve;
cp -rf ./data/Uniswap/hourly_liquidity_full.parquet ./5.\ Functional\ PCA\ analysis\ of\ the\ liquidity\ curve;
cp -rf ./data/Uniswap/hourly_liquidity_pricecentered_full.parquet ./5.\ Functional\ PCA\ analysis\ of\ the\ liquidity\ curve;

cp -rf ./data/Uniswap/hourly_pool_state_full.parquet ./6.\ Legendre\ basis\ decomposition;
cp -rf ./data/Uniswap/hourly_liquidity_full.parquet ./6.\ Legendre\ basis\ decomposition;
cp -rf ./data/Uniswap/hourly_liquidity_pricecentered_full.parquet ./6.\ Legendre\ basis\ decomposition;

cp -rf ./data/Uniswap/hourly_liquidity_full.parquet ./7.\ Gegenbauer\ Polynomials;

rm -rf ./data;


