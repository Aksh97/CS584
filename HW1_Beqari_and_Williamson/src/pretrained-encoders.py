from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

pandarallel.initialize()

# read in data 
# warning: pandas ignores bad rows
train_labels = []
train_comments = []
valid_labels = ["+1", "-1"]
with open('trainhw1.txt') as file_reader:
    lines = file_reader.readlines()
    for index, line in enumerate(lines):
        labeled_review = line.strip().split("\t")
        label = labeled_review[0].strip()
        if label not in valid_labels:
            label = "<SKIP>"
        try:
            comment = labeled_review[1]
        except IndexError:
            label = "<SKIP>"
            comment = "" 
        train_labels.append(label)
        train_comments.append(comment)

test_comments = []
with open('testdatahw1.txt') as file_reader:
    lines = file_reader.readlines()
    for line in lines:
        test_comments.append(line.strip())

train_dict = {"comment": train_comments, "label": train_labels}
train_data = pd.DataFrame(train_dict, columns=["comment", "label"]) 

test_dict = {"comment": train_comments}
x_test = pd.DataFrame(test_comments, columns=["comment"]) 

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

# embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
embedder = SentenceTransformer('bert-large-nli-mean-tokens')

# retrieve embeddings from model
x_train["embeddings"] = x_train["comment"].parallel_apply(lambda row: embedder.encode(row))
x_eval["embeddings"]  = x_eval["comment"].parallel_apply(lambda row: embedder.encode(row))
x_test["embeddings"]  = x_test["comment"].parallel_apply(lambda row: embedder.encode(row))

# save dataframes to file
x_train.to_pickle("x_train.pkl")
x_eval.to_pickle("x_eval.pkl")
x_test.to_pickle("x_test.pkl")
