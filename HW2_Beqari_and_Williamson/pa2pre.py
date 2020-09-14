#    name: pa2pre.py
# purpose: Student's add code to preprocessing of the data

# Recall that any preprocessing you do on your training
# data, you must also do on any future data you want to
# predict.  This file allows you to perform any
# preprocessing you need on my undisclosed test data

NB_CLASSES=10
from tensorflow import keras
import numpy as np

def processTestData(X, y):

    train_data   = np.load('MNISTXtrain1.npy')
    train_labels = np.load('MNISTytrain1.npy')

    eval_data    = np.load('MNIST_X_test_1.npy')
    eval_labels  = np.load('MNIST_y_test_1.npy')

    # X preprocessing goes here -- students optionally complete

    # y preprocessing goes here.  y_test becomes a ohe
    y_ohe = keras.utils.to_categorical (y, NB_CLASSES)
    return X, y_ohe