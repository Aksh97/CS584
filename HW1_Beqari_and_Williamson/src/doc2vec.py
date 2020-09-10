from gensim.summarization.textcleaner import clean_text_by_word
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import multiprocessing

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

def paragraph_to_tokens(paragraph):
    try:
        text_dict = clean_text_by_word(paragraph)
        tokens = [value.token for key, value in text_dict.items()]
        return tokens
    except TypeError:
        print('warning: nan found in paragraph - {}'.format(paragraph))
    return []

# step 1: convert paragraph to tokens
x_train["tokens"] = x_train["comment"].apply(lambda row: paragraph_to_tokens(row))
x_eval["tokens"]  = x_eval["comment"].apply(lambda row: paragraph_to_tokens(row))
x_test["tokens"]  = x_test["comment"].apply(lambda row: paragraph_to_tokens(row))
    
# *** experimental: train the vocabolary with all words ***
all_tokens = pd.concat([x_train, x_eval, x_test], keys=['tokens'], names=['tokens'])

# doc2vec model
# docs: https://radimrehurek.com/gensim/models/doc2vec.html
cores = multiprocessing.cpu_count()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_tokens["tokens"].tolist())]
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x_train["tokens"].tolist())]
model_d2v = Doc2Vec(
    documents, 
    dm=0, 
    vector_size=300, 
    negative=5, 
    hs=0, 
    sample=0, 
    window=3, 
    min_count=2, 
    workers=cores, 
    alpha=0.005, 
    min_alpha=0.001,
    dbow_words=1, # 0-bow, 1-skip gram 
    epochs=50)

# step 2: retrieve embeddings from model
x_train["embeddings"] = x_train["tokens"].apply(lambda row: model_d2v.infer_vector(row))
x_eval["embeddings"]  = x_eval["tokens"].apply(lambda row: model_d2v.infer_vector(row))
x_test["embeddings"]  = x_test["tokens"].apply(lambda row: model_d2v.infer_vector(row))

x_train.to_pickle("x_train.pkl")
x_eval.to_pickle("x_eval.pkl")
x_test.to_pickle("x_test.pkl")
