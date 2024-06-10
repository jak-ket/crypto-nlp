import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def get_sentiment_score(df: pd.DataFrame, lag: str, exponential_decay: bool, num_comments_weighting: bool, alpha_name: str):
    """lag can be specified as timedelta"""

    def get_rolling(s: pd.Series, exponential_decay=exponential_decay, lag=lag, times=df["datetime"]):
        if exponential_decay:
            return s.ewm(halflife=lag, times=times).sum()
        else:
            return s.rolling(lag).sum()

    if num_comments_weighting:
        df["rolling_positive_score"] = get_rolling(df["num_comments"]*df["positive_score"])
        df["rolling_negative_score"] = get_rolling(df["num_comments"]*df["negative_score"])
    else:
        df["rolling_positive_score"] = get_rolling(df["positive_score"])
        df["rolling_negative_score"] = get_rolling(df["negative_score"])
    
    df[alpha_name] = (df["rolling_positive_score"] - df["rolling_negative_score"]) / (df["rolling_positive_score"] + df["rolling_negative_score"] + 1e-4)

    return df


def plot_realization(df: pd.DataFrame, alpha: str, perfs: List[str], norm: str = "l1", threshold: float = 0.0):

    sns.set_style('whitegrid')
    sns.set_palette("Set2")
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for perf in perfs:
        
        if norm == "l1":
            df["real"] = np.sign(df[alpha]) * df[perf]
            df["real"] = df["real"] * (np.abs(df[alpha]) > threshold)
        elif norm == "l2":
            beta = df[alpha].cov(df[perf]) / df[alpha].var()
            df["real"] = beta * df[alpha] * df[perf]
        
        # Calculate cumulative sum for the 'real' column
        df['cumulative_real'] = df['real'].cumsum()
        
        # Plot each performance with a label
        sns.lineplot(data=df, x="datetime", y="cumulative_real", ax=ax, label=f"{perf} ({norm} norm)")
    
    # Adding titles and labels
    ax.set_title("Cumulative Realized Performance")
    ax.set_ylabel("Cumulative Performance")
    ax.set_xlabel("Date")
    ax.legend(title="Performance Metrics")
    
    return fig, ax

    
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

        risk_free_rate = 0
        annualized_mean_excess_return = annualized_mean_return - risk_free_rate

        sharpe_ratio = annualized_mean_excess_return / annualized_std_dev
        
        metrics["sharpe"].append(sharpe_ratio)
        metrics["bias"].append(bias)
        metrics["beta"].append(beta)
        
        
    return metrics
    
    
