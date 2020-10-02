import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, kernel_size=4, activation="relu", strides=1, padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(32, kernel_size=4, activation="relu", strides=1, padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(784, activation="tanh")(x)

x = layers.Reshape(target_shape=(28, 28, 1))(x)
x = layers.Conv2D(32, kernel_size=4, activation="relu", strides=1, padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2D(64, kernel_size=4, activation="relu", strides=1, padding="same")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x),

model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
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

history = model.fit(X_train, y_train, 
    batch_size=128, 
    epochs=30,
    validation_data=(X_test, y_test),
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
