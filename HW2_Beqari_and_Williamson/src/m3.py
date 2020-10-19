from pa2framework import MnistTwoBranchCNNClassifier, graph_report
from tensorflow import keras
from pa2pre import processTestData
import numpy as np
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

def main():
    np.random.seed(1671)
    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    # (X_train, y_train) = processTestData(X_train, y_train)

    print('KERA modeling build starting...')
    ## Build your model here

    model = MnistTwoBranchCNNClassifier(epochs=60, batch_size=128)
    history = model.fit(X_train, y_train, verbose=1).get_history()

    # used for model evaluation 
    # X_test = np.load("MNIST_X_test_1.npy")
    # X_test = X_test.astype("float32") / 255
    # X_test = np.expand_dims(X_test, -1)

    # y_test = np.load("MNIST_y_test_1.npy")
    # y_test = keras.utils.to_categorical(y_test, 10)

    # used for model evaluation and visualition 
    # history = model.fit(X_train, y_train, validation_data=([X_test,X_test], y_test), verbose=1).get_history()
    # graph_report(model, history, X_test_file="MNIST_X_test_1.npy", y_test_file="MNIST_y_test_1.npy")

    # save your model
    model.save_model(parms.outModelFile)

if __name__ == '__main__':
    main()
