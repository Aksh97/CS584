import numpy as np
from pa2framework import MnistDNNClassifier, MnistCNNClassifier, NestedCrossValidation

# load train data 
x_train = np.load("MNISTXtrain1.npy")
y_train = np.load("MNISTytrain1.npy")

# load test data
x_test = np.load("MNIST_X_test_1.npy")
y_test = np.load("MNIST_y_test_1.npy")

model = MnistDNNClassifier()
params = {
    "epochs": [5, 10, 15, 20], 
    "batch_size": [26, 32, 64, 128],
    "nodes": [[128, 64, 32], [128, 32, 64], [64, 128, 32]],
    "activation": ['relu', 'sigmoid', 'tanh'],
    "dropout": [0.2, 0.5]
    }
model = MnistDNNClassifier()
cv = NestedCrossValidation(model, params, x_train,  y_train, x_test, y_test)
cv.fit()

model = MnistCNNClassifier()
params = {
    "epochs": [5, 10, 15, 20], 
    "batch_size": [26, 32, 64, 128],
    "filters": [[32, 64], [64, 32], [128, 64], [64, 128]],
    "kernel": [3, 4, 5],
    "pool": [2, 3],
    "strategy": ['pool', 'norm'],
    "dropout": [0.2, 0.5]
    }
cv = NestedCrossValidation(model, params, x_train,  y_train, x_test, y_test)
cv.fit()