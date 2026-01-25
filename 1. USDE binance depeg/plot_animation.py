import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.dates as mdates


def plot_klines(df= None, close = None, low = None, high = None, open = None, ax = None, label = None):
    if df is not None:
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            open = df['open']
            date = df.index
        except KeyError:
            raise ValueError("Dataframe must have 'high', 'low', 'close', 'open' columns.")
    else:
        if high is None or low is None or close is None or open is None:
            raise ValueError("Provide either a dataframe or all of high, low, close, and open data.")

    d = pd.Series(date).diff().dropna()
    step = d.median() if len(d) else pd.Timedelta(minutes=1)
    width = step * 0.8  # 80% of the candle spacing
    height = close - open
    bottom = np.where(height > 0, open, close + abs(height))
    color = np.where(height > 0, 'green', 'red')
    if ax is None :
        plt.bar(date, height, bottom=bottom, color=color,edgecolor = color, align='center', width=width, label=label)
        plt.vlines(date, ymin=low, ymax=high, color=color, linewidth=1)
    else :
        ax.bar(date, height, bottom=bottom, color=color, align='center', width=width, label=label)
        ax.vlines(date, ymin=low, ymax=high, color=color, linewidth=1)



if __name__ == "__main__":
    
    #Load binance USDE minute OHLCV data and Uniswap block USDE price data
    bin = pd.read_parquet('usde_minute_binance.parquet')
    uni = pd.read_parquet('uniswap_usde.parquet')
    uni = uni[["timestamp", "token1Price"]].set_index("timestamp")
    uni.token1Price = uni.token1Price.astype(float)
    uni.index = pd.to_datetime(uni.index, unit='s')

    # Center around depeg event
    center = 45250
    window = 200
    k_start, k_stop, k_step = 2, 400, 4

    # Fixed x-axis limits for ALL frames (so boundaries don't change)
    x0 = bin.index[center - window]
    x1 = bin.index[center - window + (k_stop - k_step)]  # last frame end index

    y0 = bin['low'].iloc[center-window : center-window + (k_stop - k_step)].min()
    y1 = bin['high'].iloc[center-window : center-window + (k_stop - k_step)].max() + 0.04


    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.subplots_adjust(bottom=0.18)  
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def frame_images():
        for k in range(k_start, k_stop, k_step):
            dfw = bin.iloc[center - window : center - window + k]

            ax.cla() 

            plot_klines(df=dfw, ax=ax, label = 'Binance USDE Price')
            ax.plot(uni[uni.index < dfw.index[-1]], color='royalblue', label='Uniswap USDE Price')

            ax.set_xlabel('Time')
            ax.set_ylabel('USDE Price (USD)')
            ax.legend(bbox_to_anchor=(0.72, -0.09), frameon=False, ncols=2)

            
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)  

            
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = Image.frombytes("RGBA", (w, h), fig.canvas.buffer_rgba())
            yield img.convert("P", palette=Image.Palette.ADAPTIVE)

    # Save GIF
    it = frame_images()
    first = next(it)
    first.save(
        "ohlc.gif",
        save_all=True,
        append_images=it,
        duration=100, 
        loop=0,
        optimize=True,
        disposal=1,
        transparency=0,
    )

    plt.close(fig)
    print('----- GIF saved as ohlc.gif -----')