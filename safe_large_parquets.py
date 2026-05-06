import pandas as pd


# First read the parquet file and then write it back with zstd compression. This is a workaround for the issue of large parquet files not being readable in some environments. By rewriting the file with zstd compression, we can ensure that it can be read without issues.
df = pd.read_parquet('4. Stablecoin liquidity ownership/3CRV_lpevents.parquet')
df.to_parquet('4. Stablecoin liquidity ownership/3CRV_lpevents.parquet', compression = 'zstd')