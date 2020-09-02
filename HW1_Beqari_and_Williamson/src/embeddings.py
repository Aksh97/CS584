from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

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

# use pretrain model to for word embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# use the first 100 words or pad 
max_paragraph_tokens = 100

def text_to_token_ids(text, max_length):

    if text is None or not text or not isinstance(text, str):
        return 

    token_ids = tokenizer(text, 
        return_token_type_ids=False, 
        return_attention_mask=False, 
        add_special_tokens=False, 
        max_length=max_length, 
        pad_to_max_length=True, 
        truncation=True)['input_ids']
    return token_ids

def token_ids_to_tokens(token_ids):
    tokens = [tokenizer.convert_ids_to_tokens(di) for di in token_ids]
    return tokens

def token_ids_to_embeddings(token_ids):

    if token_ids is None or len(token_ids) == 0:
        # return 100.0 * np.ones(max_paragraph_tokens)
        return np.array([])

    input_ids = tf.constant(token_ids)[None, :]
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states.numpy().squeeze().flatten()

# step 1: convert text to token_ids
x_train["token_ids"] = x_train["comment"].apply(lambda row: text_to_token_ids(row, max_paragraph_tokens))
x_eval["token_ids"]  = x_eval["comment"].apply(lambda row: text_to_token_ids(row, max_paragraph_tokens))
x_test["token_ids"]  = x_test["comment"].apply(lambda row: text_to_token_ids(row, max_paragraph_tokens))

# step 2: convert token_ids to embeddings
x_train["embeddings"] = x_train["token_ids"].apply(lambda row: token_ids_to_embeddings(row))
x_eval["embeddings"]  = x_eval["token_ids"].apply(lambda row: token_ids_to_embeddings(row))
x_test["embeddings"]  = x_test["token_ids"].apply(lambda row: token_ids_to_embeddings(row))

x_train.to_pickle("x_train.pkl")
x_eval.to_pickle("x_eval.pkl")
x_test.to_pickle("x_test.pkl")
