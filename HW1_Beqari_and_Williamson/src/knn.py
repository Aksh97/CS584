from statistics import mode
from sklearn.metrics import accuracy_score
from pandarallel import pandarallel
import pandas as pd
import numpy as np
import time

pandarallel.initialize()

# read in the data
x_train = pd.read_pickle("x_train.pkl", compression="infer")
x_eval  = pd.read_pickle("x_eval.pkl", compression="infer")
x_test  = pd.read_pickle("x_test.pkl", compression="infer")

# *** knn from scratch ***
def euclidean_distance(x, y):
    return np.linalg.norm(x-y)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def print_bad_predictions(row):
    if row["label"] != row["prediction"]:
        print("label: {}, prediction: {}, comment: {}".format(row["label"], row["prediction"], row["comment"]))

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
    # prediction = mode(knn_labels)
    prediction = max(set(knn_labels), key=knn_labels.count)
    return prediction
    
def run_trials(start_k, end_k, step_k):
    accuracy_tracker = 0
    k_tracker = 0

    for k in range(start_k, end_k, step_k):
        start = time.time()
        # step 3: evaluate
        x_eval["prediction"] = x_eval["embeddings"].parallel_apply(lambda row: knn(x_train[["embeddings", "label"]].values.tolist(), row, k))
        accuracy = accuracy_score(x_eval["label"].tolist(), x_eval["prediction"].tolist())
        end = time.time()

        if accuracy > accuracy_tracker: 
            accuracy_tracker = accuracy
            k_tracker = k

        print("current k: {}, best k: {}, best accuracy: {}, elapsed time: {}".format(k, k_tracker, accuracy_tracker, end - start))

run_trials(5, 125, 5)

# view bad predictions
# x_eval.apply(lambda row: print_bad_predictions(row), axis=1, result_type="broadcast")

# step 4: predict
# uncomment code below
# num_neighbors = 30
# x_test["prediction"] = x_test["embeddings"].apply(lambda row: knn(x_train[["embeddings", "label"]].values.tolist(), row, num_neighbors))
# x_test["prediction"].to_csv(r'predictions_beqari_and_williamson.txt', header=None, index=None)
