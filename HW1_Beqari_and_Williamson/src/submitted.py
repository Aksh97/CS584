
import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def cosine_similarity(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def knn(rows, row, num_neighbors):
    distance_and_label = []
    # calculate distance and keep track of labels
    for target_row in rows:
        train_embeddings = np.array(target_row[0])
        test_embeddings = np.array(row)
        # distance = euclidean_distance(train_embeddings, test_embeddings)
        distance = cosine_similarity(train_embeddings, test_embeddings)
        label = target_row[1]
        distance_and_label.append([distance, label])
    # sort the list by distance positional element
    # save only the k-th specified neigbors
    # change reverse=False for euclidean distance
    sorted_distance_and_label = sorted(distance_and_label, key=lambda x: x[0], reverse=True)[:num_neighbors]  
    # return the mode of the neighbors 
    knn_labels = [element[1] for element in sorted_distance_and_label]
    prediction = max(set(knn_labels), key=knn_labels.count)
    return prediction

# read in data line by line 
# because pandas ignores bad rows
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
x_train = pd.DataFrame(train_dict, columns=["comment", "label"]) 

test_dict = {"comment": test_comments} # <-train_comments
x_test = pd.DataFrame(test_dict, columns=["comment"]) 

# this line will download the first time
# may require restart of the env
embedder = SentenceTransformer('bert-large-nli-mean-tokens')
print("embedding paragraphs - please wait this make take a while...")

# retrieve embeddings from model
x_train["embeddings"] = x_train["comment"].apply(lambda row: embedder.encode(row))
x_test["embeddings"]  = x_test["comment"].apply(lambda row: embedder.encode(row))
print("running knn - almost there now...")

best_num_neighbors = 56 # according to cross-validation
x_test["prediction"] = x_test["embeddings"].apply(lambda row: knn(x_train[["embeddings", "label"]].values.tolist(), row, best_num_neighbors))

# sometimes adds additional blank line at the end
# behavior is not always the same
print("writting to file... num lines: {}".format(len(x_test["prediction"].values.tolist())))
x_test["prediction"].to_csv(r'labelssubmitted.txt', header=None, index=None)
