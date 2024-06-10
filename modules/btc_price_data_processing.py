import pandas as pd
from typing import List


def load_btc_price_data():
    
    df = pd.read_csv("data_raw/bitcoin_2015_to_2023.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date":"datetime", "Open":"open"})
    
    return df

def restrict_datetime_range(df: pd.DataFrame, btc_df: pd.DataFrame):
    
    start_date = df["datetime"].min()
    end_date = df["datetime"].max()
    
    btc_df = btc_df[(btc_df["datetime"] >= start_date) & (btc_df["datetime"] <= end_date)]
    
    return btc_df

def add_performance_metrics(df: pd.DataFrame, lags: List[int]):
    
    btc_df = load_btc_price_data()
    btc_df = restrict_datetime_range(df, btc_df)
    
    freq = "h"
    dfh = btc_df.groupby(pd.Grouper(key="datetime", freq=freq))["open"].first().to_frame().copy()   

    for lag in lags:
        dfh.loc[:,f"open_lag_{lag}"] = dfh["open"].shift(periods=-lag)
        dfh[f"perf_{lag}"] = (dfh[f"open_lag_{lag}"] - dfh["open"]) / dfh["open"]
        
            
        df = pd.merge_asof(dfh[f"perf_{lag}"], df, on='datetime', direction='backward')
        
    return df
    

