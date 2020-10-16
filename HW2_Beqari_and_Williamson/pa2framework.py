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


class MnistVAETransformer():
    
    def __init__(self, model, batch):
        self._model = model
        self._batch = batch

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = np.expand_dims(x, -1).astype("float32") / 255
        x = np.reshape(x, (self._batch, 28, 28, 1))
        x = self._model.predict(x)
        return x


class MnistDNNClassifier():  

    def __init__(self, epochs=15, batch_size=128, nodes=[128, 64, 32], activation="relu", dropout=0.5):
        self._model = None
        self._history = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._nodes = nodes
        self._activation = activation
        self._dropout = dropout
        self._num_classes = 10
        self._input_shapes = (28, 28, 1)
        self._accuracy = keras.metrics.CategoricalAccuracy()

    def fit(self, x, y=None, validation_split=None, verbose=0):

        # normalize images to the [0, 1] range
        x_train = x.astype("float32") / 255

        # extend one dim to shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)

        # one-hot encoding
        y_train = keras.utils.to_categorical(y, self._num_classes)

        self._model = keras.Sequential(
            [
                keras.Input(shape=self._input_shapes),
                layers.Flatten(),
                layers.Dense(self._nodes[0], activation="relu"),
                layers.Dropout(self._dropout),
                layers.Dense(self._nodes[1], activation="relu"),
                layers.Dropout(self._dropout),
                layers.Dense(self._nodes[2], activation=self._activation),
                layers.Dropout(self._dropout),
                layers.Dense(self._num_classes, activation="softmax"),
            ]
        )

        self._model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self._history = self._model.fit(x_train, y_train, 
            batch_size=self._batch_size, 
            epochs=self._epochs, 
            validation_split=validation_split,
            verbose=verbose,
            workers=-1)
        return self

    def get_history(self): 
        return self._history

    def predict(self, x):
        x_predict = x.astype("float32") / 255
        x_predict = np.expand_dims(x_predict, -1)
        prediction = self._model.predict(x_predict)
        prediction = np.argmax(prediction, axis=1)
        return(prediction)

    def score(self, x, y=None):
        x_eval = x.astype("float32") / 255
        x_eval = np.expand_dims(x_eval, -1)
        y_true = keras.utils.to_categorical(y, self._num_classes)
        score = self._model.evaluate(x_eval, y_true, verbose=0)
        return(score[1])

    def get_params(self, deep=True):
        return {
            "epochs": self._epochs, 
            "batch_size": self._batch_size,
            "nodes": self._nodes,
            "activation" : self._activation,
            "dropout": self._dropout
            }

    def set_params(self, epochs, batch_size, nodes, activation, dropout):
        self._epochs = epochs
        self._batch_size = batch_size
        self._nodes = nodes
        self._activation = activation
        self._dropout = dropout
        return self 

    def get_model(self):
        return self._model

    def save_model(self, path):
        self._model.save(path)
 
class MnistCNNClassifier():  

    def __init__(self, epochs=15, batch_size=128, filters=[32, 64], kernel=3, pool=2, strategy='pool', dropout=0.5):
        self._model = None
        self._history = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._filters=filters
        self._kernel=kernel
        self._pool=pool
        self._strategy=strategy
        self._dropout = dropout
        self._num_classes = 10
        self._input_shapes = (28, 28, 1)
        self._accuracy = keras.metrics.CategoricalAccuracy()

    def _add_strategy(self, layers):
        if self._strategy == "pool":
            return layers.MaxPooling2D(pool_size=self._pool)
        elif self._strategy == "norm":
            return layers.BatchNormalization()
        
    def fit(self, x, y=None, validation_split=None, validation_data=None, verbose=0):
        # normalize images to the [0, 1] range
        x_train = x.astype("float32") / 255

        # extend one dim to shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)

        # one-hot encoding
        y_train = keras.utils.to_categorical(y, self._num_classes)

        self._model = keras.Sequential(
            [
                keras.Input(shape=self._input_shapes),
                layers.Conv2D(self._filters[0], kernel_size=self._kernel, activation="relu"),
                self._add_strategy(layers),
                layers.Conv2D(self._filters[1], kernel_size=self._kernel, activation="relu"),
                self._add_strategy(layers),
                layers.Flatten(),
                layers.Dropout(self._dropout),
                layers.Dense(self._num_classes, activation="softmax"),
            ]
        )

        self._model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self._history = self._model.fit(x_train, y_train, 
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
        prediction = self._model.predict(x_predict)
        prediction = np.argmax(prediction, axis=1)
        return(prediction)

    def score(self, x, y=None):
        x_eval = x.astype("float32") / 255
        x_eval = np.expand_dims(x_eval, -1)
        y_true = keras.utils.to_categorical(y, self._num_classes)
        score = self._model.evaluate(x_eval, y_true, verbose=0)
        return(score[1])

    def get_params(self, deep=True):
        return {
            "epochs": self._epochs, 
            "batch_size": self._batch_size,
            "filters": self._filters,
            "kernel": self._kernel,
            "pool": self._pool,
            "strategy": self._strategy,
            "dropout": self._dropout
            }

    def set_params(self, epochs, batch_size, filters, kernel, pool, strategy, dropout):
        self._epochs = epochs
        self._batch_size = batch_size
        self._filters = filters
        self._kernel = kernel
        self._pool = pool
        self._strategy = strategy
        self._dropout = dropout
        return self 

    def save_model(self, path):
        self._model.save(path)

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
        prediction = self._model.predict(x_predict)
        prediction = np.argmax(prediction, axis=1)
        return(prediction)

    def score(self, x, y=None):
        x_eval = x.astype("float32") / 255
        x_eval = np.expand_dims(x_eval, -1)
        y_true = keras.utils.to_categorical(y, self._num_classes)
        score = self._model.evaluate(x_eval, y_true, verbose=0)
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


# add multiple filters
# source: https://stackoverflow.com/questions/57438922/different-size-filters-in-the-same-layer-with-tensorflow-2-0

class NestedCrossValidation():  

    def __init__(self, model, params, x_train, y_train, x_eval, y_eval):
        self._model = model
        self._params = params
        self._x_train = x_train
        self._y_train = y_train
        self._x_eval = x_eval
        self._y_eval = y_eval

    def fit(self):
    
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

	    # define search
        search = GridSearchCV(estimator=self._model, param_grid=self._params, cv=cv, refit=True, n_jobs=1)

	    # execute search
        result = search.fit(self._x_train, self._y_train)

	    # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
	    # evaluate model on the hold out dataset
        yhat = best_model.predict(self._x_eval)

	    # evaluate the model
        accuracy = keras.metrics.Accuracy()
        accuracy.update_state(self._y_eval, yhat)
 
	    # report progress
        print('test-set accuracy: {:.10f}, train-set best score: {:.10f}, best params: {}'.format(accuracy.result().numpy(), result.best_score_, result.best_params_))

        return result.best_params_

def print_greyscale(pixels_1, pixels_2, width=28, height=28):
    
    def get_single_greyscale(pixel):
        val = 232 + np.round(pixel * 23)
        return '\x1b[48;5;{}m \x1b[0m'.format(int(val))

    for l in range(height):
        line_pixels = np.concatenate((pixels_1[l * width:(l+1) * width], pixels_2[l * width:(l+1) * width]), axis=None)
        print(''.join(get_single_greyscale(p) for p in line_pixels))
