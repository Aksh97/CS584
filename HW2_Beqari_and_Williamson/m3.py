import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pa2pre import processTestData
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

    input1 = keras.Input(shape=(28, 28, 1))
    x1 = layers.Conv2D(32, kernel_size=3, activation="relu", strides=1, padding="same")(input1)
    x1 = layers.MaxPooling2D(pool_size=2)(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Conv2D(64, kernel_size=3, activation="relu", strides=1, padding="same")(x1)
    x1 = layers.MaxPooling2D(pool_size=2)(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Conv2D(128, kernel_size=3, activation="relu", strides=1, padding="same")(x1)
    x1 = layers.MaxPooling2D(pool_size=2)(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Flatten()(x1)

    input2 = keras.Input(shape=(28, 28, 1))
    x2 = layers.Conv2D(32, kernel_size=4, activation="relu", strides=1, padding="same")(input2)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Conv2D(64, kernel_size=4, activation="relu", strides=1, padding="same")(x2)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Conv2D(128, kernel_size=4, activation="relu", strides=1, padding="same")(x2)
    x2 = layers.MaxPooling2D(pool_size=2)(x2)
    x2 = layers.Dropout(0.5)(x2)
    x2 = layers.Flatten()(x2)

    x = layers.Concatenate(axis=1)([x1, x2])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(784, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Reshape(target_shape=(28, 28, 1))(x)
    x = layers.Conv2D(32, kernel_size=3, activation="relu", strides=1, padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(64, kernel_size=3, activation="relu", strides=1, padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x),

    model = keras.Model(inputs=[input1, input2], outputs=outputs, name="experimental")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    X_train = np.load("MNISTXtrain1.npy")
    X_train = X_train.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)

    y_train = np.load("MNISTytrain1.npy")
    y_train = keras.utils.to_categorical(y_train, 10)

    X_test = np.load("MNIST_X_test_1.npy")
    X_test = X_test.astype("float32") / 255
    X_test = np.expand_dims(X_test, -1)

    y_test = np.load("MNIST_y_test_1.npy")
    y_test = keras.utils.to_categorical(y_test, 10)

    history = model.fit([X_train, X_train], y_train, 
    batch_size=128, 
    epochs=60,
    validation_data=([X_test, X_test], y_test),
    verbose=1,
    workers=-1)

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

if __name__ == '__main__':
    main()
