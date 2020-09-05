from statistics import mode
import pandas as pd
import numpy as np
# import os
# os.remove("./dummy.pkl")

# https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d#:~:text=2.,and%20train%20for%2030%20epochs.
# import os

# read in the data
x_train = pd.read_pickle("x_train.pkl");
x_eval  = pd.read_pickle("x_eval.pkl");
x_test  = pd.read_pickle("x_test.pkl");

### *** knn from scratch ***

num_neighbors = 9

def euclidean_distance(x, y):
    return np.linalg.norm(x-y)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def knn(rows, row, num_neighbors):
    distance_and_label = []
    # calculate distance and keep track of labels
    for target_row in rows:
        trained_embeddings = np.array(target_row[0])
        predicted_embeddings = np.array(row)
        distance = cosine_similarity(trained_embeddings, predicted_embeddings)
        label = target_row[1]
        distance_and_label.append([distance, label])
    # sort the list by distance positional element
    # save only the k-th specified neigbors
    sorted_distance_and_label = sorted(distance_and_label, key=lambda x: x[0], reverse=True)[:num_neighbors]
    # return the mode of the neighbors 
    knn_labels = [element[1] for element in sorted_distance_and_label]
    return mode(knn_labels)

# step 3: evaluate
x_eval["prediction"] = x_eval["embeddings"].apply(lambda row: knn(x_train[["embeddings", "label"]].values.tolist(), row, num_neighbors))
print(x_eval["prediction"].head(5))
x_eval.to_csv('baseline-eval.csv', index=False)
