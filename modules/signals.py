import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from modules.btc_price_data_processing import load_btc_price_data

def get_sentiment_score(df: pd.DataFrame, lag: str, exponential_decay: bool, num_comments_weighting: bool, alpha_name: str):
    """lag can be specified as timedelta"""

    def get_rolling(s: pd.Series, exponential_decay=exponential_decay, lag=lag, times=df["datetime"]):
        if exponential_decay:
            return s.ewm(halflife=lag, times=times).mean()
        else:
            return s.rolling(window=lag).mean()

    if num_comments_weighting:
        df["rolling_positive_score"] = get_rolling(df["num_comments"]*df["positive_score"])
        df["rolling_negative_score"] = get_rolling(df["num_comments"]*df["negative_score"])
    else:
        df["rolling_positive_score"] = get_rolling(df[["positive_score"]])
        df["rolling_negative_score"] = get_rolling(df[["negative_score"]])
    
    df[alpha_name] = (df["rolling_positive_score"] - df["rolling_negative_score"]) / (df["rolling_positive_score"] + df["rolling_negative_score"] + 1e-4)

    return df
    

def get_relative_strength_index(lookback:int=24*14, lower_cutoff:str=None, upper_cutoff:str=None):
    """"
    Relative Strength Index (RSI)
    lookback: number of hours to look back
    lower_cutoff: date where signal time series should start
    """
    # get bitcoin time series
    btc_df = load_btc_price_data()
    # btc_df = restrict_datetime_range(df, btc_df)

    freq = "h"
    dfh = btc_df.groupby(pd.Grouper(key="datetime", freq=freq))["open"].first().to_frame().copy()   

    # cutoff time series
    if lower_cutoff:
        dfh = dfh.loc[dfh.index>=lower_cutoff]
    
    if upper_cutoff:
        dfh = dfh.loc[dfh.index<=upper_cutoff]

    # get hourly returns
    dfh.loc[:,f"open_lag_1"] = dfh["open"].shift(periods=-1)
    dfh[f"perf_1"] = (dfh[f"open_lag_1"] - dfh["open"]) / dfh["open"]
    
    # get extra columns with indicator for positive and negative returns
    dfh["is_gain"] = np.where(dfh[f"perf_1"] > 0, 1, 0)
    dfh["is_loss"] = np.where(dfh[f"perf_1"] < 0, 1, 0)

    # rolling mean over gains and losses
    dfh["avg_gain"] = dfh["is_gain"].rolling(f"{lookback}h").sum() / lookback 
    dfh["avg_loss"] = dfh["is_loss"].rolling(f"{lookback}h").sum() / lookback

    # compute rsi
    # dfh["rsi"] = 100 - (100/(1+dfh["rs_raw"]))
    dfh["rsi"] = dfh["avg_gain"] / (dfh["avg_gain"] + dfh["avg_loss"])

    # compute signal
    def get_rsi_signal(row):
        if row["rsi"] > 0.55:
            return -1
        elif row["rsi"] < 0.45:
            return +1
        else:
            return 0

    dfh["rsi_signal"] = dfh.apply(get_rsi_signal, axis=1)   
    
    return dfh


def get_moving_average_crossover(sma:int=24, lma:int=24*7, thres:float=0, lower_cutoff:str=None, upper_cutoff:str=None) -> pd.DataFrame:
    """
    Create Moving Average CrossOver (MACO) signal
    lower_cutoff: date where signal time series should start

    sma: short moving average in hours
    lma: long moving average in hours
    thres: threshold for crossover signal, e.g. thres=0.2 requires sMA > 1.2*lMA for signal = +1 or sMA < 0.8*lMA for signal = -1
    """
    # get bitcoin time series
    btc_df = load_btc_price_data()
    # btc_df = restrict_datetime_range(df, btc_df)

    freq = "h"
    dfh = btc_df.groupby(pd.Grouper(key="datetime", freq=freq))["open"].first().to_frame().copy()   

    # cutoff time series
    if lower_cutoff:
        dfh = dfh.loc[dfh.index>=lower_cutoff]

    if upper_cutoff:
        dfh = dfh.loc[dfh.index<=upper_cutoff]

    # get moving averages
    dfh["sMA"] = dfh["open"].rolling(f"{sma}h").mean()
    dfh["lMA"] = dfh["open"].rolling(f"{lma}h").mean()

    # get signal
    def get_maco_signal(row):
        if row["sMA"] > (1+thres)*row["lMA"]:
            return +1
        elif row["sMA"] < (1-thres)*row["lMA"]:
            return -1
        else:
            return 0
    dfh["maco_signal"] = dfh.apply(get_maco_signal, axis=1)

    return dfh
    

def plot_realization(df: pd.DataFrame, alphas: List[str], perfs: List[str], 
                     norm: str = "l1", threshold: float = 0.0, save: bool = False,
                     alpha_labels: Optional[List[str]] = None, perf_labels: Optional[List[str]] = None):
    
    # Also allow for single strings
    if isinstance(alphas, str):
        alphas = [alphas]
    if isinstance(perfs, str):
        perfs = [perfs]

    # Use default labels if custom labels are not provided
    if alpha_labels is None:
        alpha_labels = alphas
    if perf_labels is None:
        perf_labels = perfs

    sns.set_style('white')
    
    # Set up colors and linestyles
    color_palette = sns.color_palette("Set2", n_colors=len(alphas))
    linestyles = ['-', '--', '-.', ':']

    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})
    
    _, ax = plt.subplots(figsize=(12, 6))

    # Dummy plots for the legend for colors (alpha labels)
    for i, label in enumerate(alpha_labels):
        ax.plot([], [], color=color_palette[i], label=label)

    # Dummy plots for the legend for linestyles (performance labels)
    if len(perfs) > 1:
        for j, label in enumerate(perf_labels):
            ax.plot([], [], linestyle=linestyles[j % len(linestyles)], color='black', label=label)

    for i, alpha in enumerate(alphas):
        for j, perf in enumerate(perfs):
            if norm == "l1":
                df["real"] = np.sign(df[alpha]) * df[perf]
                df["real"] = df["real"] * (np.abs(df[alpha]) > threshold)
            elif norm == "l2":
                beta = df[alpha].cov(df[perf]) / df[alpha].var()
                df["real"] = df[alpha] * (df[perf] * beta)
            
            # Calculate cumulative sum for the 'real' column
            df['cumulative_real'] = df['real'].cumsum()
            
            # Actual plot for each combination
            sns.lineplot(
                data=df, 
                x="datetime", 
                y="cumulative_real", 
                ax=ax, 
                color=color_palette[i],
                linestyle=linestyles[j % len(linestyles)]
            )
    
    # Set labels and configure the legend
    ax.set_ylabel("Cumulative Realization")
    ax.set_xlabel("Date")
    
    # Organize the legend to first show alpha colors, then performance linestyles (if more than one)
    handles, labels = ax.get_legend_handles_labels()
    # Include performance labels in the legend only if there are more than one perf
    if len(perfs) > 1:
        order = list(range(len(alpha_labels))) + list(range(len(alpha_labels), len(alpha_labels) + len(perf_labels)))
    else:
        order = list(range(len(alpha_labels)))

    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title="Legend", loc='upper left')
    
    sns.despine()
    
    if save:
        plt.savefig("realized_performance.pdf", bbox_inches='tight')
    
    plt.show()

    
def compute_key_metrics(df: pd.DataFrame, alpha: str, perfs: List[str]):
    
    metrics = {
        "sharpe": [],
        "bias": [],
        "beta": []
    }
    
    for perf in perfs:
        
        beta = df[alpha].cov(df[perf]) / df[alpha].var()
        bias = df[alpha].cov(df[perf]) / df[alpha].std()
        
        df["realized"] = beta * df[alpha] * df[perf]
        
        mean_return_hourly = df['realized'].mean()
        std_dev_hourly = df['realized'].std()

        annualized_mean_return = mean_return_hourly * 8760  # 365 days * 24 hours
        annualized_std_dev = std_dev_hourly * np.sqrt(8760)

        risk_free_rate = 0.0
        annualized_mean_excess_return = annualized_mean_return - risk_free_rate

        sharpe_ratio = annualized_mean_excess_return / annualized_std_dev
        
        metrics["sharpe"].append(sharpe_ratio)
        metrics["bias"].append(bias)
        metrics["beta"].append(beta)
        
        
    return metrics
    
    
