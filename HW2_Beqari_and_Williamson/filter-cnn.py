import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
# from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import mean, std
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

class MnistCNNClassifier():  

    def __init__(self, epochs=15):
        self._model = None
        self._epochs = epochs
        self._batch_size = 128
        self._num_classes = 10
        self._input_shapes = (28, 28, 1)
        self._accuracy = keras.metrics.CategoricalAccuracy()
        
    def fit(self, x, y=None):
        # normalize images to the [0, 1] range
        x_train = x.astype("float32") / 255

        # extend one dim to shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)

        # one-hot encoding
        y_train = keras.utils.to_categorical(y, self._num_classes)

        inputs = keras.Input(shape=(32, 32, 3), name="img")

        # ***
        x1 = layers.Conv2D(filters=32, kernel_size=3)(inputs)
        x2 = layers.Conv2D(filters=16, kernel_size=5)(inputs)
        #match dimensions (height and width) of x1 or x2 here 
        x3 = layers.Concatenate(axis=-1)[x1,x2]
        # ***

        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        block_1_output = layers.MaxPooling2D(3)(x)

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_2_output = layers.add([x, block_1_output])

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        block_3_output = layers.add([x, block_2_output])

        x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
        x = layers.GlobalAveragePooling2D()(x)  
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10)(x)

        model = keras.Model(inputs, outputs, name="toy_resnet")
        model.summary()

        return self

    def predict(self, x):
        x_predict = x.astype("float32") / 255
        x_predict = np.expand_dims(x_predict, -1)
        prediction = self._model.predict(x_predict)
        prediction = np.argmax(prediction, axis=1)
        return(prediction)

    def score(self, x, y=None):
        x_eval = x.astype("float32") / 255
        x_eval = np.expand_dims(x, -1)
        y_true = keras.utils.to_categorical(y, self._num_classes)
        score = self._model.evaluate(x_eval, y_true, verbose=0)
        return(score[1])

    def get_params(self, deep=True):
        return {"epochs": self._epochs}

    def set_params(self, epochs):
        self._epochs = epochs
        return self 

    # save to .h5
    def save(self, path):
        self._model.save(path)

    # load from .h5
    def load(self, path):
        self._model = keras.models.load_model(path)