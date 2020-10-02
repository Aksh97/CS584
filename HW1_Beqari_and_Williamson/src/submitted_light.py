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

# read in the data
x_train = pd.read_pickle("x_train.pkl", compression="infer")
x_test  = pd.read_pickle("x_test.pkl", compression="infer")
print("running knn - almost there now...")

best_num_neighbors = 56 # according to cross-validation
x_test["prediction"] = x_test["embeddings"].apply(lambda row: knn(x_train[["embeddings", "label"]].values.tolist(), row, best_num_neighbors))
x_test["prediction"].to_csv(r'labelssubmitted-light.txt', header=None, index=None)