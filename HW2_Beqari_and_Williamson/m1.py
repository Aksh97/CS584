from pa2framework import MnistDNNClassifier
from sklearn.metrics import classification_report
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

    # (X_train, y_train) = processTestData(X_train, y_train)

    print('KERA modeling build starting...')
    ## Build your model here

    model = MnistDNNClassifier(
        epochs=20, 
        batch_size=26, 
        nodes=[128, 64, 32], 
        activation='sigmoid', 
        dropout=0.2)
    
    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1).get_history()

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