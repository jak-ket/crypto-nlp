import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")


def create_lda_model(
    df: pd.DataFrame, n_topics: int, text_column: str = "processed_text"
):
    """
    Create an LDA model for topic modeling.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        n_topics (int): The number of topics to generate.
        text_column (str, optional): The name of the column containing the text data. Defaults to "processed_text".

    Returns:
        Tuple[pd.DataFrame, gensim.models.ldamodel.LdaModel, gensim.corpora.MmCorpus, gensim.corpora.Dictionary]:
            - df (pd.DataFrame): The input DataFrame with the processed text column.
            - lda (gensim.models.ldamodel.LdaModel): The trained LDA model.
            - corpus (gensim.corpora.MmCorpus): The corpus representation of the text data.
            - dictionary (gensim.corpora.Dictionary): The dictionary mapping of the text data.
    """
    
    # Preprocess the text
    df["processed_text"] = df[text_column].apply(_strip_down_string)

    # Create LDA model
    df, lda, corpus, dictionary = _lda_topics(df, n_topics, text_column)

    return df, lda, corpus, dictionary


def _strip_down_string(text: str):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join words back into one string separated by space
    return " ".join(words)


def _lda_topics(dataframe, k, text_column="processed_text"):
    # Tokenizer to split words
    tokenizer = RegexpTokenizer(r"\w+")

    # Tokenize documents
    texts = [tokenizer.tokenize(doc.lower()) for doc in dataframe[text_column]]

    # Create a Gensim Dictionary from the texts
    dictionary = corpora.Dictionary(texts)

    # Filter out extremes to limit the vocabulary
    dictionary.filter_extremes(no_below=10, no_above=0.1, keep_n=10000)

    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count).
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA model
    lda = models.LdaModel(
        corpus=corpus, num_topics=k, id2word=dictionary, passes=10, random_state=42
    )

    # Assign topics and their probabilities
    topic_probabilities = []

    for doc in corpus:
        # Get the topic distribution for the document
        doc_topics = lda.get_document_topics(doc, minimum_probability=0)
        doc_prob_dict = dict(doc_topics)
        # Fill in missing probabilities if any topic has probability 0
        topic_probs = [doc_prob_dict.get(i, 0) for i in range(k)]
        topic_probabilities.append(topic_probs)

    # Creating columns for each topic's probability
    for i in range(k):
        dataframe[f"topic_{i}_prob"] = [probs[i] for probs in topic_probabilities]

    # Assign the most probable topic to a separate column
    dataframe["topic"] = dataframe.apply(
        lambda row: max((row[f"topic_{i}_prob"], i) for i in range(k))[1], axis=1
    )

    return dataframe, lda, corpus, dictionary
