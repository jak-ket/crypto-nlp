import pandas as pd


def load_reddit_data():
    df =  pd.read_csv("data_2022/submission.csv")
    return df


def load_bitcoin_data():
    return pd.read_csv("data_2022/bitcoin_hourly.csv")