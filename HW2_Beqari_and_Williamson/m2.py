from pa2framework import MnistCNNClassifier
from tensorflow import keras
import matplotlib.pyplot as plt
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

    X_test = np.load("MNIST_X_test_1.npy")
    X_test = X_test.astype("float32") / 255
    X_test = np.expand_dims(X_test, -1)

    y_test = np.load("MNIST_y_test_1.npy")
    y_test = keras.utils.to_categorical(y_test, 10)

    # (X_train, y_train) = processTestData(X_train, y_train)

    print('KERA modeling build starting...')
    ## Build your model here

    model = MnistCNNClassifier(
        epochs=50, 
        batch_size=128,
        filters=[64, 128],
        kernel=4,
        pool=2,
        strategy='pool',
        dropout=0.5)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1).get_history()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    ## save your model
    # model.save_model("mnist_dnn_model")

if __name__ == '__main__':
    main()