import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
from tensorflow import keras
from tensorflow.keras import layers
from pa2pre import processTestData
import numpy as np
import argparse

class MnistTwoBranchCNNClassifier():  

    def __init__(self, epochs=15, batch_size=128):
        self._model = None
        self._history = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._num_classes = 10
        self._input_shapes = (28, 28, 1)
        self._accuracy = keras.metrics.CategoricalAccuracy()

    def fit(self, x, y=None, validation_split=None, validation_data=None, verbose=0):
        # normalize images to the [0, 1] range
        x_train = x.astype("float32") / 255
        # extend one dim to shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        # one-hot encoding
        y_train = keras.utils.to_categorical(y, self._num_classes)

        input1 = keras.Input(shape=self._input_shapes)
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

        input2 = keras.Input(shape=self._input_shapes)
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

        self._model = keras.Model(inputs=[input1, input2], outputs=outputs, name="experimental")
        self._model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self._history = self._model.fit([x_train, x_train], y_train, 
            batch_size=self._batch_size, 
            epochs=self._epochs, 
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=verbose,
            workers=-1)
        return self

    def get_history(self): 
        return self._history

    def predict(self, x):
        x_predict = x.astype("float32") / 255
        x_predict = np.expand_dims(x_predict, -1)
        prediction = self._model.predict_on_batch(x=[x_predict, x_predict])[0]
        prediction = np.argmax(prediction, axis=1)
        return(prediction)

    def score(self, x, y=None):
        x_eval = x.astype("float32") / 255
        x_eval = np.expand_dims(x_eval, -1)
        y_true = keras.utils.to_categorical(y, self._num_classes)
        score = self._model.evaluate([x_eval, x_eval], y_true, verbose=0)
        return(score[1])

    def get_params(self, deep=True):
        return {
            "epochs": self._epochs, 
            "batch_size": self._batch_size,
            }

    def set_params(self, epochs, batch_size):
        self._epochs = epochs
        self._batch_size = batch_size
        return self 

    def save_model(self, path):
        self._model.save(path)

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

    # save your model
    model.save_model(parms.outModelFile)

if __name__ == '__main__':
    main()
