# from transformers import pipeline
import pandas as pd
from tqdm import tqdm
# import transformers
# import torch
import csv
import os

def get_roberta_checkpoint():
    df = pd.read_csv("bitcoin_2022/sentiment_roberta.csv")
    id_cols = ["submission"]
    sentiment_cols = ["positive_score", "negative_score"]
    return df[id_cols + sentiment_cols]


def get_llm_checkpoint(model_id: str):
    model_id = model_id.replace("/", "-")
    df = pd.read_csv(f"bitcoin_2022/sentiment_results_{model_id}.csv")
    id_cols = ["selftext"]
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


# def get_sentiment_llm(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    
    
#     pipeline = transformers.pipeline(
#         "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
#     )
    
#     system_prompt = """
#     Categorize the provided reddit post as "positive" if the post is positive about bitcoin, "negative" if it is negative about bitcoin. If none of these categories are applicable or it is unclear, please respond with "none/unclear". Respond briefly with "positive", "negative", or "none/unclear".
#     """
    
#     model_id = model_id.replace("/", "-")
    
#     # Check if the CSV file already exists to handle the header
#     file_exists = os.path.isfile(f'sentiment_results_{model_id}.csv')
    
#     # Open a file to append
#     with open(f'bitcoin_2022/sentiment_results_{model_id}.csv', mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
        
#         # Write the header if the file does not exist
#         if not file_exists:
#             writer.writerow(['selftext', 'sentiment_label', 'positive_score', 'negative_score'])
    
#         # Iterate through each post in the DataFrame
#         for _, row in tqdm(df.iterrows(), total=df.shape[0]):
#             # Extract post text
#             post_text = row['selftext']
            
#             # Perform sentiment analysis
#             try:
#                 prompt = "Post: " + post_text
            
#                 messages = [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt},
#                 ]

#                 terminators = [
#                     pipeline.tokenizer.eos_token_id,
#                     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#                 ]
                
#                 outputs = pipeline(
#                     messages,
#                     max_new_tokens=256,
#                     eos_token_id=terminators,
#                     do_sample=True,
#                     temperature=0.6,
#                     top_p=0.9,
#                 )
                
#                 positive_score = 0
#                 negative_score = 0
#                 sentiment_label = outputs[0]["generated_text"][-1]["content"]
#                 if sentiment_label == "positive":
#                     positive_score = 1
#                 elif sentiment_label == "negative":
#                     negative_score = 1
        
#             except Exception as e:
#                 print(f"Error processing text: {e}")
#                 sentiment_label, positive_score, negative_score = 'ERROR', 0, 0
            
#             # Write the results to the CSV file
#             writer.writerow([post_text, sentiment_label, positive_score, negative_score])
        