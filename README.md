![cover image](/images/cover.png)

# Extracting Bitcoin price signals from Reddit posts
Our research leverages NLP techniques applied to posts from Reddit's r/Bitcoin subreddit to build a signal for predicting forward returns of the Bitcoin price (BTC).

## Signal Creation Pipeline
The notebook `sentiment_signal.ipynb` contains the full pipeline from data cleaning to signal creation and performance metrics.

Contains the signal creation pipeline including
- loading and pre-processing of reddit corpus
- computation or checkpoint loading of sentiment scores using RoBERTa
- filtering of topics using LDA 
- creation of the sentiment signal 
- computation of baseline strategies
- creation of lagged hourly bitcoin return time series
- performance plots of cumulative realization

Contains experiments including
- can the feature num_comments improve the signal?
- is $\beta$ stable over time?

## Exploration and Experiments
To produce the streamlined pipeline notebook, we conducted research and  experiments in the notebooks starting with `explore_`
- `explore_benchmark_signals` develops and tests the RSI and MACO benchmark signals
- `explore_bitcoin_price` explores the BTC price in 2022 and creates hourly dataframe
- `explore_ema_multivariate_signal` develops and tests the signal based on an exponential smoothing of the sentiment score
- `explore_lda_topics` applies an LDA topic model to discover the trading topic and produces the file *assets/topic_keywords.csv*
- `explore_reddit_corpus` produces summary statistics for the reddit corpus and explores first sentiment techniques
- `explore_sentiment_analysis` tests different techniques for topic filtering and sentiment analysis, including the production of human evaluation tables
- `explore_sentiment_gpt4` uses GPT4 to produce sentiment labels in order to compare them with RoBERTa and human performance

## Folders
- The /modules folder contains functionality derived from the exploration file.
- The /assets folder contains flatfiles derived from the exploration notebooks.
- The /data_2022 folder contains reddit and bitcoin data for 2022
- The /human_eval folder contains files with reddit posts with human and LLM sentiment label.

## Installation
`pip install -r requirements.txt`