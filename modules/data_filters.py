import pandas as pd

def filter_lda_trading_topic(df, text_col, topic_file, topic_name):
    topic_keywords = pd.read_csv(topic_file)
    trading_keywords = topic_keywords.loc[topic_keywords["Topic"]==topic_name, "Keyword"].values
    lda_filter = df[text_col].str.lower().str.contains("|".join(trading_keywords))
    df = df.loc[lda_filter] # filter on LDA topics
    return df