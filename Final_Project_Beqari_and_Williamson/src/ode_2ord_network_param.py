import os
from os import replace

from tensorflow.python.keras.backend import dtype
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

class ODENetwork():

    def __init__(self):
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.param_opt = tf.keras.optimizers.Adam(learning_rate=0.005) #, clipnorm=1.0
        # self.param_opt = keras.optimizers.SGD(learning_rate=0.05)
        self.history = []
        self.t_index, self.t_train = self._data()
        self.x_exact               = self._exact()
        self.t_observed, self.x_observed = self._observed()

        self.b = tf.Variable(10, dtype=tf.double)
        self.k = tf.Variable(32, dtype=tf.double)
        self.b_history = []
        self.k_history = []

        # losses 
        self.huber_loss = tf.keras.losses.Huber()

    def _data(self):
        # t_state = np.array(np.linspace(0.0, 1, num=1000), dtype=np.uint32)
        t_train = np.arange(0, 1, 0.001)
        t_index = np.arange(len(t_train))
        return t_index, t_train

    def _exact(self):
        x_exact = np.asarray([-3.0 * t * np.exp(-7 * t) for t in self.t_train])
        return x_exact

    def _observed(self):
        mu, sigma = 0, 0.005
        rand_index = np.random.choice(self.t_index, 200, replace=False)
        t_observed = self.t_train[rand_index]
        x_random   = np.random.normal(mu, sigma, len(t_observed))
        x_observed = np.asarray([x_random[i] + self.x_exact[j] for i, j in enumerate(rand_index)])
        return t_observed, x_observed

    def gaussian(self, x, beta=2):
        return tf.keras.backend.exp(-tf.keras.backend.pow(beta * x, 2))

    # def create_model(self):
    #     inputs = keras.Input(shape=(1,))
    #     l1 = layers.Dense(2048, activation=self.gaussian)(inputs)
    #     outputs = layers.Dense(1, activation="linear")(l1)
    #     self.model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
    #     print(self.model.summary())
    #     return self

    def create_model(self):
        inputs = keras.Input(shape=(1,))
        l1 = layers.Dense(20, activation=self.gaussian)(inputs)
        l2 = layers.Dense(20, activation=self.gaussian)(l1)
        l3 = layers.Dense(20, activation="sigmoid")(l2)
        outputs = layers.Dense(1, activation="linear")(l3)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
        print(self.model.summary())
        return self

    @tf.function
    def loss_f(self, f, f_t, f_tt):
        loss_f = tf.keras.backend.square(f_tt + self.b * f_t + self.k * f)
        return loss_f

    @tf.function
    def loss_ic(self, t_0):
        ic_t = tf.constant(3, dtype=tf.double)
        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(t_0)
            f_0 = tf.cast(self.model(t_0, training=False), dtype=tf.double)
            f_t0 = tape_ord_1.gradient(f_0, t_0)
            loss_ic = tf.keras.backend.square(f_0) + tf.keras.backend.square(f_t0 + ic_t)
        return loss_ic

    @tf.function
    def apply_training_step(self, t, t_0):
        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(t)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(t)
                tape_ord_1.watch(self.b)
                tape_ord_1.watch(self.k)

                # f_training = self.model(t, training=True)
                f = tf.cast(self.model(t, training=False), dtype=tf.double)
                f_t = tape_ord_1.gradient(f, t)
                f_tt = tape_ord_2.gradient(f_t, t)
                loss_f = tf.keras.backend.map_fn(
                    lambda x: self.loss_f(x[0], x[1], x[2]),
                    (f, f_t, f_tt),
                    dtype=tf.double)
                loss_f = tf.cast(tf.reduce_mean(loss_f), dtype=tf.double)

                u = tf.cast(self.model(self.t_observed, training=False), dtype=tf.double)
                loss_u = tf.cast(self.huber_loss(self.x_observed, u), dtype=tf.double)

                loss_ic = tf.cast(tf.reduce_mean(self.loss_ic(t_0)), dtype=tf.double)

                loss = loss_u + loss_f + loss_ic

        grad_b = tape_ord_1.gradient(loss, self.b)
        self.param_opt.apply_gradients(zip([grad_b], [self.b]))
        grad_k = tape_ord_1.gradient(loss, self.k)
        self.param_opt.apply_gradients(zip([grad_k], [self.k]))

        grads = tape_ord_1.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights), experimental_aggregate_gradients=False)
        return loss

    def train(self):
        epochs = 10000
        batch_size = 25 # len(self.t_train)
        t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)

        stop_training = False

        for epoch in range(epochs):

            train_dataset = tf.data.Dataset.from_tensor_slices(self.t_train)
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

            if stop_training: break

            for step, t in enumerate(train_dataset):
                self.b_history.append(self.b.numpy())
                self.k_history.append(self.k.numpy())
                loss = self.apply_training_step(t, t_0)

                if step % 100 == 0:
                    self.history.append(tf.reduce_mean(loss))
                    print("Training loss for step/epoch {}/{}: {}".format(step, epoch, tf.reduce_mean(loss)))
                    if(tf.reduce_mean(loss).numpy() <= 0.0001): 
                        stop_training = True
                        break

    def predict(self):
        predictions = self.model.predict(self.t_train)
        return predictions

    def get_history(self):
        return self.history

def main():
    ode_net = ODENetwork()
    ode_net.create_model()
    ode_net.train()

    print("b: {}, k: {}".format(ode_net.b, ode_net.k))

    plt.plot(ode_net.get_history())
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    fig_a = plt.figure()
    ax1 = fig_a.add_subplot(111)
    ax1.scatter(ode_net.t_observed, ode_net.x_observed, s=10, c='red', marker="o", label='sample')
    ax1.scatter(ode_net.t_train, ode_net.x_exact, s=10, c='blue', marker="s", label='exact')
    ax1.scatter(ode_net.t_train, ode_net.predict(), s=10, c='orange', marker="s", label='approx.')
    plt.legend(loc='lower right')
    plt.show()

    fig_b = plt.figure()
    ax2 = fig_b.add_subplot(111)
    ax2.scatter(ode_net.b_history, ode_net.k_history, s=10, c='blue', marker="x", label='gd')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
