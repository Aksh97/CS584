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
#   x" + bx' + kx = 0
#   x(0)  = 0
#   x'(0) = -3
#   x(t) = −3te^−7t

class CustomCallback(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):

        t = logs["t"]
        x = logs["x"]

        self.model.axs[0].clear()
        self.model.fig.suptitle("b={}, k={}".format(round(logs["b"], 3), round(logs["k"], 3)))
        # self.model.axs[0].title("b={}, k={}".format(round(logs["b"], 3), round(logs["k"], 3)))
        self.model.axs[0].scatter([0.0, 0.0, 1.0, 1.0], [0.1, -0.25, 0.1, -0.25], c='red', marker="+")
        self.model.axs[0].scatter(t, x)
        self.model.axs[0].scatter(logs["t_train"], self.model.predict(logs["t_train"]))
        plt.pause(0.05)

    def on_epoch_end(self, epoch, logs=None):

        self.model.axs[1].scatter(epoch, logs['loss'])
        self.model.axs[1].scatter(epoch, logs['loss_f'])
        self.model.axs[1].scatter(epoch, logs['loss_ic'])
        self.model.axs[1].scatter(epoch, logs['loss_u'])
        # plt.title('Model Loss')
        # self.model.axs[1].ylabel('loss')
        # self.model.axs[1].xlabel('epoch')
        # self.model.axs[1].ylim(0, 2)
        self.model.axs[1].legend(['total', 'f', 'ic', 'u'], loc='upper right')
        plt.pause(0.05)



class ODENetwork(tf.keras.Model):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    # param_opt = tf.keras.optimizers.SGD(learning_rate=0.05)

    t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)
    ic_t = tf.constant(3, dtype=tf.double)
    b = tf.Variable(5, dtype=tf.double)
    k = tf.Variable(10, dtype=tf.double)

    loss_tracker  = tf.keras.metrics.Mean(name="loss")
    loss_f_trckr  = tf.keras.metrics.Mean(name="loss_f")
    loss_ic_trckr = tf.keras.metrics.Mean(name="loss_ic")
    loss_u_trckr  = tf.keras.metrics.Mean(name="loss_u")

    fig, axs = plt.subplots(2)
    # plt.autoscale(enable=False)
    plt.show(block=False)
    plt.pause(2)

    t_train = np.arange(0, 1, 0.0001)
    
    @staticmethod 
    def data(min, max, dt):
        t_train = np.arange(min, max, dt)
        t_index = np.arange(len(t_train))
        return t_index, t_train

    @staticmethod 
    def exact(t_train):
        x_exact = np.asarray([-3.0 * t * np.exp(-7 * t) for t in t_train])
        return x_exact

    @staticmethod 
    def observed(t_index, t_train, x_exact, mu=0, sigma=0.05, points=200):
        rand_index = np.sort(np.random.choice(t_index, points, replace=False), axis=None)
        t_observed = t_train[rand_index]
        x_random   = np.random.normal(mu, sigma, len(t_observed))
        x_observed = np.asarray([x_random[i] + x_exact[j] for i, j in enumerate(rand_index)])
        return t_observed, x_observed

    @tf.function
    def gaussian(x, beta=2):
        return tf.keras.backend.exp(-tf.keras.backend.pow(beta * x, 2))

    @tf.function
    def loss_total(self, u, f_t, f_tt, u_exact):
        loss_f = tf.keras.backend.square(f_tt + self.b * f_t + self.k * u)
        loss_u = tf.keras.backend.square(u_exact - u)
        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(self.t_0)
            f_0 = tf.cast(self(self.t_0, training=False), dtype=tf.double)
            f_t0 = tape_ord_1.gradient(f_0, self.t_0)
            loss_ic = tf.keras.backend.square(f_0) + tf.keras.backend.square(f_t0 + self.ic_t)
            loss_ic = tf.reshape(loss_ic, (1,))
        self.loss_f_trckr.update_state(loss_f)
        self.loss_ic_trckr.update_state(loss_ic)
        self.loss_u_trckr.update_state(loss_u)
        return loss_f + loss_ic + loss_u

    @tf.function
    def loss_f(self, f, f_t, f_tt):
        loss_f = tf.keras.backend.square(f_tt + self.b * f_t + self.k * f)
        return loss_f

    @tf.function
    def loss_ic(self):
        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(self.t_0)
            f_0 = tf.cast(self(self.t_0, training=False), dtype=tf.double)
            f_t0 = tape_ord_1.gradient(f_0, self.t_0)
            loss_ic = tf.keras.backend.square(f_0) + tf.keras.backend.square(f_t0 + self.ic_t)
            loss_ic = tf.reshape(loss_ic, (1,))
        return loss_ic

    @tf.function
    def loss_u(self, u_pred, u_exact):
        loss_u = tf.keras.backend.square(u_exact - u_pred)
        return loss_u

    def train_step(self, data):
        t_observed, x_observed = data
        x_observed = tf.cast(tf.expand_dims(x_observed, axis=-1), dtype=tf.double)
        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(t_observed)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(t_observed)
                tape_ord_1.watch(self.b)
                tape_ord_1.watch(self.k)

                u = tf.cast(self(t_observed, training=False), dtype=tf.double)
                f_t = tape_ord_1.gradient(u, t_observed)
                f_tt = tape_ord_2.gradient(f_t, t_observed)

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss_total(x[0], x[1], x[2], x[3]),
                    (u, f_t, f_tt, x_observed),
                    dtype=tf.double)
                loss = tf.reduce_mean(loss, axis=-1)

        grad_b = tape_ord_1.gradient(loss, self.b)
        grad_k = tape_ord_1.gradient(loss, self.k)
        self.optimizer.apply_gradients(zip([grad_b], [self.b]))
        self.optimizer.apply_gradients(zip([grad_k], [self.k])) 

        grads = tape_ord_1.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return { 
            "loss": self.loss_tracker.result(), 
            "loss_f": self.loss_f_trckr.result(),
            "loss_ic": self.loss_ic_trckr.result(), 
            "loss_u": self.loss_u_trckr.result(),
            "t": t_observed,
            "x": tf.keras.backend.squeeze(x_observed, axis=1),
            "b": self.b,
            "k": self.k,
            "t_train": self.t_train}
    
    @property
    def metrics(self):
        return [
            self.loss_tracker, 
            self.loss_f_trckr, 
            self.loss_ic_trckr, 
            self.loss_u_trckr]

def main():

    t_index, t_train       = ODENetwork.data(0, 1, 0.0001)
    x_exact                = ODENetwork.exact(t_train)
    t_observed, x_observed = ODENetwork.observed(t_index, t_train, x_exact, mu=0, sigma=0.05, points=2000)

    batch_size = 5 # len(t_train)
    epochs     = 5000 # 100000

    inputs = keras.Input(shape=(1,))
    l1 = layers.Dense(50, activation="sigmoid")(inputs)
    l2 = layers.Dense(50, activation="sigmoid")(l1)
    l3 = layers.Dense(50, activation="sigmoid")(l2)
    outputs = layers.Dense(1, activation="linear")(l3)
    model = ODENetwork(inputs, outputs)

    print(model.summary())
    model.compile()

    t_obs_ds = tf.data.Dataset.from_tensor_slices(t_observed)
    t_obs_ds = t_obs_ds.map(lambda x: tf.cast(x, dtype=tf.double))
    t_obs_ds = t_obs_ds.shuffle(len(t_observed), seed=123)
    t_obs_ds = t_obs_ds.batch(batch_size)

    x_obs_ds = tf.data.Dataset.from_tensor_slices(x_observed)
    x_obs_ds = x_obs_ds.map(lambda x: tf.cast(x, dtype=tf.double))
    x_obs_ds = x_obs_ds.shuffle(len(t_observed), seed=123)
    x_obs_ds = x_obs_ds.batch(batch_size)

    diffq_ds = tf.data.Dataset.zip((t_obs_ds, x_obs_ds))
    diffq_ds = diffq_ds.cache()

    history = model.fit(
        diffq_ds,
        epochs=epochs, 
        workers=-1,
        verbose=0,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), CustomCallback()])

    print("\n b: {}, k: {}".format(model.b.numpy(), model.k.numpy()))

    plt.plot(history.history['loss'])
    plt.plot(history.history['loss_f'])
    plt.plot(history.history['loss_ic'])
    plt.plot(history.history['loss_u'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 2)
    plt.legend(['total', 'f', 'ic', 'u'], loc='upper right')
    plt.show()

    fig_a = plt.figure()
    ax1 = fig_a.add_subplot(111)
    ax1.scatter(t_observed, x_observed, s=10, c='red', marker="o", label='sample')
    ax1.scatter(t_train, x_exact, s=10, c='blue', marker="s", label='exact')
    ax1.scatter(t_train, model.predict(t_train), s=10, c='orange', marker="s", label='approx.')
    plt.legend(loc='lower right')
    plt.show()

    model.save('ode2ord_paramfit.h5')

if __name__ == '__main__':
    main()
