import pandas as pd
import re


def remove_urls_from_string(text: str):
    # Regular expression to identify URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Replace URLs with an empty string
    return url_pattern.sub(r'', text)


def remove_url(df: pd.DataFrame, text_column: str):
    df[text_column] = df[text_column].apply(remove_urls_from_string)
    return df

def preprocess_data(df: pd.DataFrame):
    # Pre-processing
    df = df.dropna(subset = "selftext")
    df = df[ df["removed"] != 1]

    # convert to datetime
    df["datetime"] = pd.to_datetime(df["created"], unit="s")\

    # Set 'datetime' column as the index
    df.set_index('datetime', inplace=True, drop = False)
    
    # Remove rBitcoinMod
    mask = df["author"] != "rBitcoinMod"
    mask = mask & (df["author"] != "SomeBrokeChump") # This guy also makes sticky thread, see https://www.reddit.com/r/Bitcoin/comments/sozb08/daily_discussion_february_10_2022/
    
    # Prepend title to selftext
    df["selftext"] = df["title"] + "\n" + df["selftext"]
    
    # remove urls
    df = remove_url(df, "selftext")
    
    return df[mask]