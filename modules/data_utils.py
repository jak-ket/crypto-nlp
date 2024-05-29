import pandas as pd
from modules.data_cleaning import preprocess_data

def load_reddit_data():
    df =  pd.read_csv("bitcoin_2022/submission.csv")
    df = preprocess_data(df)
    return df

def load_bitcoin_data():
    return pd.read_csv("bitcoin_2022/bitcoin_hourly.csv")