from statistics import mode
import pandas as pd
import numpy as np


### *** TODO: knn from scratch ***

def euclidean_distance(x, y):
    # x = np.array(row1) 
    # y = np.array(row2)
    # diff   = np.subtract(x, y) 
    # diffsq = np.square(diff)
    # sumsq  = np.sum(diffsq)
    # return np.sqrt(sumsq)
    return np.linalg.norm(x-y)

def cosine_similarity(x, y):
    # x = np.array(row1) 
    # y = np.array(row2)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def get_knn(df, new_df, num_neighbors, distance_func=euclidean_distance):
    # df = df.copy(deep=True)
    new_row = new_df["embeddings"].values
    df["distance"] = df["embeddings"].apply(lambda row: distance_func(row, new_row))
    df = df.sort_values(by="distance", ascending=True, kind='quicksort')
    knn = df["distance", "label"].head(num_neighbors).values.tolist()
    return knn

# step 3: 
# x_train = pd.read_json('x_train.json', orient='split')
# x_eval  = pd.read_json('x_eval.json', orient='split')
# x_test  = pd.read_json('x_test.json', orient='split')

# x_train = pd.read_feather(x_train, use_threads=True);
# x_eval  = pd.read_feather(x_eval, use_threads=True);
# x_test  = pd.read_feather(x_test, use_threads=True);

x_train = pd.read_pickle("x_train.pkl");
x_eval  = pd.read_pickle("x_eval.pkl");
# x_test  = pd.read_pickle("x_test.pkl");

num_neighbors = 8
# test = get_knn(x_train, x_eval[['embeddings']].head(1), num_neighbors)
# print("test: ", x_eval['embeddings'].head(1))

# x_train["embeddings"].head(5).apply(lambda row: print(type(row)))
# x_train.head(5).apply(lambda row: print(row))

x_train = x_train.drop(['comment', 'token_ids'], axis=1)
x_eval  = x_eval.drop(['comment', 'token_ids'], axis=1)

test = get_knn(x_train, x_eval.head(1), num_neighbors) 
print(test)

# pd.read_pickle("./dummy.pkl")
# import os
# os.remove("./dummy.pkl")