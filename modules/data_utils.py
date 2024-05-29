import pandas as pd

def load_reddit_data():
    return pd.read_csv("bitcoin_2022/submission.csv")

def load_bitcoin_data():
    return pd.read_csv("bitcoin_2022/bitcoin_hourly.csv")