import pandas as pd

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


