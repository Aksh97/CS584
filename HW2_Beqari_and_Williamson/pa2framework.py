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

class MnistDNNClassifier():  

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

        self._model = keras.Sequential(
            [
                keras.Input(shape=self._input_shapes),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(32, activation="sigmoid"),
                layers.Dropout(0.5),
                layers.Dense(self._num_classes, activation="softmax"),
            ]
        )

        self._model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self._model.fit(x_train, y_train, batch_size=self._batch_size, epochs=self._epochs, verbose=0) # workers=-1
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

        self._model = keras.Sequential(
            [
                keras.Input(shape=self._input_shapes),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self._num_classes, activation="softmax"),
            ]
        )

        self._model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self._model.fit(x_train, y_train, batch_size=self._batch_size, epochs=self._epochs, verbose=0, workers=-1)
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

# *** Cross validation with grid search ***

class NestedCrossValidation():  
    def __init__(self):
        self._x = np.load('MNISTXtrain1.npy')
        self._y = np.load('MNISTytrain1.npy')

    def run(self):
    
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=0)
        outer_results = list()

        for train_index, eval_index in cv_outer.split(self._x):

	        # split into test and eval
            x_train, x_eval = self._x[train_index, :], self._x[eval_index, :]
            y_train, y_eval = self._y[train_index],    self._y[eval_index]

            cv_inner = KFold(n_splits=3, shuffle=True, random_state=0)

    	    # model
            model = MnistDNNClassifier()   

	        # parameter space
            params = {"epochs" : [10]}

	        # define search
            search = GridSearchCV(model, params, cv=cv_inner, refit=True, n_jobs=1)

	        # execute search
            result = search.fit(x_train, y_train)

	        # get the best performing model fit on the whole training set
            best_model = result.best_estimator_

	        # evaluate model on the hold out dataset
            yhat = best_model.predict(x_eval)

	        # evaluate the model
            accuracy = keras.metrics.Accuracy()
            accuracy.update_state(y_eval, yhat)

            # store the result
            outer_results.append(accuracy.result().numpy())
    
	        # report progress
            print('eval-set accuracy: {:.3f}, train-set best score: {:.3f}, best params: {}'.format(accuracy.result().numpy(), result.best_score_, result.best_params_))

        # summarize the estimated performance of the model
        print('cv accuracy: mean = {:.3f}, and std. = {:.3f}'.format(mean(outer_results), std(outer_results)))

cv = NestedCrossValidation()
cv.run()

# siamese
# https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d