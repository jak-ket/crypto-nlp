# from transformers import pipeline
import pandas as pd

def get_roberta_checkpoint():
    df = pd.read_csv("bitcoin_2022/sentiment_roberta.csv")
    id_cols = ["submission"]
    sentiment_cols = ["positive_score", "negative_score"]
    return df[id_cols + sentiment_cols]


# def get_roberta_sentiment(df):
#     MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
#     sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

#     def remove_special_chars(text):
#         return ''.join(e for e in text if e.isalnum() or e.isspace())

#     sentiments_label = []
#     sentiments_score = []
#     for txt in tqdm(df["selftext"].values):
#         txt = remove_special_chars(txt)
#         try:
#             sentiment = sentiment_task(txt)
#             sentiments_label.append(sentiment[0]["label"])
#             sentiments_score.append(sentiment[0]["score"])
#         except Exception as e:
#             sentiments_label.append("N/A")
#             sentiments_score.append(0)

#     df["sentiment_label"] = sentiments_label
#     df["sentiment_score"] = sentiments_score


#     mask_positive = df["sentiment_label"] == "positive"
#     mask_negative = df["sentiment_label"] == "negative"

#     df["positive_score"] = 0
#     df.loc[mask_positive, 'positive_score'] = df[mask_positive]["sentiment_score"].values
#     df["negative_score"] = 0
#     df.loc[mask_negative, 'negative_score'] = df[mask_negative]["sentiment_score"].values
