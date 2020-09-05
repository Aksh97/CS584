from gensim.summarization.textcleaner import clean_text_by_word
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import multiprocessing
# import os

# source: https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92

# read in data
train_data = pd.read_csv('trainhw1.txt', sep="\t", names=["label", "comment"])
x_test     = pd.read_csv('testdatahw1.txt', names=["comment"])

# split test and eval
x = train_data["comment"]
y = train_data["label"]
x_train, x_eval, y_train, y_eval = train_test_split(x, y, 
    test_size=0.2, 
    random_state=0, 
    stratify=train_data["label"])

# keep labels in the same set
x_train = x_train.to_frame()
x_train["label"] = y_train

x_eval = x_eval.to_frame()
x_eval["label"] = y_eval

def paragraph_to_tokens(paragraph):
    try:
        text_dict = clean_text_by_word(paragraph)
        tokens = [value.token for key, value in text_dict.items()]
        return tokens
    except TypeError:
        print("error: nan found in paragraph - ", paragraph)
    return []

# step 1: convert paragraph to tokens
x_train["tokens"] = x_train["comment"].apply(lambda row: paragraph_to_tokens(row))
x_eval["tokens"]  = x_eval["comment"].apply(lambda row: paragraph_to_tokens(row))
x_test["tokens"]  = x_test["comment"].apply(lambda row: paragraph_to_tokens(row))
    
# *** experimental: train the vocabolary with all words ***
all_tokens = pd.concat([x_train["tokens"], x_eval["tokens"], x_test["tokens"]])

# doc2vec model
# docs: https://radimrehurek.com/gensim/models/doc2vec.html
cores = multiprocessing.cpu_count()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_tokens["tokens"].tolist())]
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x_train["tokens"].tolist())]
model_d2v = Doc2Vec(documents, vector_size=5, window=3, min_count=1, workers=cores, dbow_words=1)

# step 2: retrieve embeddings from model
x_train["embeddings"] = x_train["tokens"].apply(lambda row: model_d2v.infer_vector(row))
x_eval["embeddings"]  = x_eval["tokens"].apply(lambda row: model_d2v.infer_vector(row))
x_test["embeddings"]  = x_test["tokens"].apply(lambda row: model_d2v.infer_vector(row))

x_train.to_pickle("x_train.pkl")
x_eval.to_pickle("x_eval.pkl")
x_test.to_pickle("x_test.pkl")

print("*** done ***")
