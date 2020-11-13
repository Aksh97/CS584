import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# example: 
#   x" + 14x' + 49x = 0
#   x(0)  = 0
#   x'(0) = -3
#   x(t) = −3te^−7t
# Ψ(t) = 0 + −3t + t^2 * x^(t)

class ODENetwork():

    def __init__(self):
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        # self.optimizer = keras.optimizers.SGD(learning_rate=0.05)
        self.history = []

    def get_data(self):
        t_state = np.linspace(0.0, 1, num=1000)
        return t_state 
    
    def gaussian(self, x, beta=2):
        return tf.keras.backend.exp(-tf.keras.backend.pow(beta * x, 2))

    def create_model(self):
        inputs = keras.Input(shape=(1,))
        l1 = layers.Dense(10, activation="sigmoid")(inputs)
        l2 = layers.Dense(10, activation="sigmoid")(l1)
        l3 = layers.Dense(10, activation="sigmoid")(l2)
        outputs = layers.Dense(1, activation="linear")(l3)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
        print(self.model.summary())
        return self

    @tf.function
    def psi_func(self, t, u):
        a = tf.constant(0, dtype=tf.double)
        b = tf.constant(-3, dtype=tf.double)
        return a + b * t + tf.keras.backend.square(t) * u

    @tf.function
    def loss(self, psi, dpsi_dt, d2psi_dt2):
        c = tf.constant(14, dtype=tf.double)
        k = tf.constant(49, dtype=tf.double)
        loss = tf.keras.backend.square(d2psi_dt2 + c * dpsi_dt + k * psi)
        return loss
    
    @tf.function
    def apply_training_step(self, t, t_0):
        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(t)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(t)
                u_training = tf.cast(self.model(t, training=True), dtype=tf.double)
                psi = self.psi_func(t, u_training)
                dpsi_dt = tape_ord_1.gradient(psi, t)
                d2psi_dt2 = tape_ord_1.gradient(dpsi_dt, t)

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss(x[0], x[1], x[2]), 
                    (psi, dpsi_dt, d2psi_dt2), 
                    dtype=tf.double)
                loss = tf.reduce_mean(loss, axis=-1)
                   
        grads = tape_ord_1.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def train(self):
        epochs = 3000
        batch_size = 25 # len(self.get_data())
        t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)
   
        for epoch in range(epochs):            
            x_train = self.get_data()
            x_train = np.reshape(x_train, (-1, 1))
            train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

            for step, t in enumerate(train_dataset):
                loss = self.apply_training_step(t, t_0)

                if step % 10 == 0:
                    self.history.append(tf.reduce_mean(loss))
                    print("Training loss for step/epoch {}/{}: {}".format(step, epoch, tf.reduce_mean(loss)))

    def predict(self):
        t = self.get_data()
        u_pred = tf.cast(self.model.predict(t), dtype=tf.double)
        return tf.keras.backend.map_fn(lambda x: self.psi_func(x[0], x[1]), (t, u_pred), dtype=tf.double)

    def get_history(self):
        return self.history

    def exact_solution(self):
        return [(-3.0 * t * np.exp(-7 * t)) for t in self.get_data()]


def main():
    ode_net = ODENetwork()
    ode_net.create_model()
    ode_net.train()

    plt.plot(ode_net.get_history())
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    plt.plot(ode_net.exact_solution())
    plt.plot(ode_net.predict())
    plt.title('ODE Solution')
    plt.ylabel('x')
    plt.xlabel('step')
    plt.legend(['exact', 'approx.'], loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
