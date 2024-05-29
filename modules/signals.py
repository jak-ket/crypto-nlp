import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def get_sentiment_score(df: pd.DataFrame, lag: str, exponential_decay: bool, num_comments_weighting: bool):
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
    
    df["alpha"] = (df["rolling_positive_score"] - df["rolling_negative_score"]) / (df["rolling_positive_score"] + df["rolling_negative_score"] + 1e-4)

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
        "perf": [],
        "sharpe": [],
        "bias": [],
        "alpha": []
    }
    
    for perf in perfs:
        
        beta = df[alpha].cov(df[perf]) / df[alpha].var()
        
        metrics["perf"].append(df[perf].sum())
        metrics["sharpe"].append(df[perf].mean() / df[perf].std())
        metrics["bias"].append(beta * (df["alpha"] * df["perf"]))
        metrics["alpha"].append(df[alpha].mean())
        
        
    return metrics
    
    
