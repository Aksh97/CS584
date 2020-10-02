import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score 
from pandarallel import pandarallel
import pandas as pd
import numpy as np
import time
import uuid

# np.__config__.show()

pandarallel.initialize()

class KNNClassifier():  

    def __init__(self, num_neighbors=5):
        self._num_neighbors = num_neighbors
        self._yhat = pd.DataFrame()
        self._ytrue = pd.DataFrame()
        self._id = uuid.uuid1() 

    @staticmethod
    def euclidean_distance(x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
    @staticmethod
    def knn(rows, row, num_neighbors):
        distance_and_label = []
        # calculate distance and keep track of labels
        for target_row in rows:
            train_embeddings = np.array(target_row[0])
            test_embeddings = np.array(row)
            # distance = euclidean_distance(train_embeddings, test_embeddings)
            distance = KNNClassifier.cosine_similarity(train_embeddings, test_embeddings)
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

    def fit(self, x, y=None):
        self._labeled_embeddings = x[["embeddings", "label"]].values.tolist()
        return self

    def score(self, x, y=None):
        ytrue = x["label"].tolist()
        yhat  = x["embeddings"].parallel_apply(lambda row: self.knn(self._labeled_embeddings, row, self._num_neighbors)).tolist()
        accuracy = accuracy_score(yhat, ytrue)
        return(accuracy)

    def get_params(self, deep=True):
        return {"num_neighbors":  self._num_neighbors}

    def set_params(self, num_neighbors):
        self._num_neighbors = num_neighbors
        return self 

def run_trials(start_k, end_k, step_k):
    accuracy_tracker = 0
    k_tracker = 0

    for k in range(start_k, end_k, step_k):
        start = time.time()
        cv = StratifiedKFold(n_splits=5)
        model = KNNClassifier(num_neighbors=k)
        results_skfold = model_selection.cross_val_score(model, x_train, x_train["label"], cv=cv, n_jobs=1)
        accuracy = results_skfold.mean()
        end = time.time()

        if accuracy > accuracy_tracker: 
            accuracy_tracker = accuracy
            k_tracker = k

        print("current k: {}, best k: {}, best accuracy: {:.3f}, elapsed time: {}".format(k, k_tracker, accuracy_tracker, end - start))
    return k_tracker

# read in the data
x_train = pd.read_pickle("x_train.pkl", compression="infer")
x_test  = pd.read_pickle("x_test.pkl", compression="infer")

best_num_neighbors = run_trials(5, 125, 1)

# predict step - uncomment code below
x_test["prediction"] = x_test["embeddings"].parallel_apply(lambda row: KNNClassifier.knn(x_train[["embeddings", "label"]].values.tolist(), row, 65))
x_test["prediction"].to_csv(r'predictions_beqari_and_williamson.txt', header=None, index=None)
