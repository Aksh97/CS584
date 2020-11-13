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

class ODENetwork(tf.keras.Model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    param_opt = tf.keras.optimizers.Adam(learning_rate=0.005) #, clipnorm=1.0

    t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)
    b = tf.Variable(10, dtype=tf.double)
    k = tf.Variable(32, dtype=tf.double)

    huber_loss = tf.keras.losses.Huber()
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    @staticmethod 
    def data():
        t_train = np.arange(0, 1, 0.001)
        t_index = np.arange(len(t_train))
        return t_index, t_train

    @staticmethod 
    def exact(t_train):
        x_exact = np.asarray([-3.0 * t * np.exp(-7 * t) for t in t_train])
        return x_exact

    @staticmethod 
    def observed(t_index, t_train, x_exact):
        mu, sigma = 0, 0.005
        rand_index = np.random.choice(t_index, 200, replace=False)
        t_observed = t_train[rand_index]
        x_random   = np.random.normal(mu, sigma, len(t_observed))
        x_observed = np.asarray([x_random[i] + x_exact[j] for i, j in enumerate(rand_index)])
        return t_observed, x_observed

    @tf.function
    def gaussian(self, x, beta=2):
        return tf.keras.backend.exp(-tf.keras.backend.pow(beta * x, 2))

    @tf.function
    def loss_f(self, f, f_t, f_tt):
        loss_f = tf.keras.backend.square(f_tt + self.b * f_t + self.k * f)
        return loss_f

    @tf.function
    def loss_ic(self, t_0):
        ic_t = tf.constant(3, dtype=tf.double)
        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(t_0)
            f_0 = tf.cast(self(t_0, training=False), dtype=tf.double)
            f_t0 = tape_ord_1.gradient(f_0, t_0)
            loss_ic = tf.keras.backend.square(f_0) + tf.keras.backend.square(f_t0 + ic_t)
        return loss_ic

    def train_step(self, data):
        t, (t_observed, x_observed) = data
        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(t)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(t)
                tape_ord_1.watch(self.b)
                tape_ord_1.watch(self.k)

                # f_training = self.model(t, training=True)
                f = tf.cast(self(t, training=False), dtype=tf.double)
                f_t = tape_ord_1.gradient(f, t)
                f_tt = tape_ord_2.gradient(f_t, t)
                loss_f = tf.keras.backend.map_fn(
                    lambda x: self.loss_f(x[0], x[1], x[2]),
                    (f, f_t, f_tt),
                    dtype=tf.double)
                loss_f = tf.cast(tf.reduce_mean(loss_f), dtype=tf.double)

                u = tf.cast(self(t_observed, training=False), dtype=tf.double)
                loss_u = tf.cast(self.huber_loss(x_observed, u), dtype=tf.double)

                loss_ic = tf.cast(tf.reduce_mean(self.loss_ic(self.t_0)), dtype=tf.double)

                loss = loss_u + loss_f + loss_ic

        grad_b = tape_ord_1.gradient(loss, self.b)
        self.param_opt.apply_gradients(zip([grad_b], [self.b]))
        grad_k = tape_ord_1.gradient(loss, self.k)
        self.param_opt.apply_gradients(zip([grad_k], [self.k]))

        grads = tape_ord_1.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights), experimental_aggregate_gradients=False)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker]

def main():
    batch_size = 25
    epochs     = 100000

    inputs = keras.Input(shape=(1,))
    l1 = layers.Dense(20, activation="sigmoid")(inputs)
    l2 = layers.Dense(20, activation="sigmoid")(l1)
    l3 = layers.Dense(20, activation="sigmoid")(l2)
    outputs = layers.Dense(1, activation="linear")(l3)
    model = ODENetwork(inputs, outputs)
    print(model.summary())

    model.compile()

    t_index, t_train       = ODENetwork.data()
    x_exact                = ODENetwork.exact(t_train)
    t_observed, x_observed = ODENetwork.observed(t_index, t_train, x_exact)

    train_dataset = tf.data.Dataset.from_tensor_slices(t_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(50)

    obs_dataset = tf.data.Dataset.from_tensor_slices((t_observed, x_observed))
    obs_dataset = obs_dataset.batch(batch_size)

    combined_dataset = tf.data.Dataset.zip((train_dataset, obs_dataset))
 
    history = model.fit(combined_dataset, epochs=epochs, workers=-1)

    print("\n b: {}, k: {}".format(model.b, model.k))

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    fig_a = plt.figure()
    ax1 = fig_a.add_subplot(111)
    ax1.scatter(t_observed, x_observed, s=10, c='red', marker="o", label='sample')
    ax1.scatter(t_train, x_exact, s=10, c='blue', marker="s", label='exact')
    ax1.scatter(t_train, model.predict(t_train), s=10, c='orange', marker="s", label='approx.')
    plt.legend(loc='lower right')
    plt.show()

    # fig_b = plt.figure()
    # ax2 = fig_b.add_subplot(111)
    # ax2.scatter(ode_net.b_history, ode_net.k_history, s=10, c='blue', marker="x", label='gd')
    # plt.legend(loc='lower right')
    # plt.show()

if __name__ == '__main__':
    main()
