import pandas as pd

def get_sentiment_score(df: pd.DataFrame, lag: str = "24h"):
    df["rolling_positive_score"] = df["positive_score"].rolling(lag).sum()
    df["rolling_negative_score"] = df["negative_score"].rolling(lag).sum()
    df["alpha"] = (df["rolling_positive_score"] - df["rolling_negative_score"]) / (df["rolling_positive_score"] + df["rolling_negative_score"] + 1e-4)
    return df

