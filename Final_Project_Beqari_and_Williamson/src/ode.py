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

class MonitorCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        t = logs["t"]
        x = logs["x"]

        t_train = logs["t_train"]
        x_exact = logs["x_exact"]

        b = logs["b"]
        k = logs["k"]
        lam = logs["lam"]

        self.model.axs[0].clear()
        self.model.fig.suptitle("Training... [ b={:.3f}, k={:.3f}, λ={:.2e} ]".format(b, k, lam))
        self.model.axs[0].scatter(t_train, x_exact, s=10, c='blue', marker="s", label='exact')
        self.model.axs[0].scatter(t, x, s=10, c='red', marker="o", label='sample')
        self.model.axs[0].scatter(t_train, self.model.predict(t_train), s=10, c='orange', marker="s", label='approx.')
        self.model.axs[0].set_xlabel('t')
        self.model.axs[0].set_ylabel('x')
        self.model.axs[0].axis([0.0, 1.0, -0.17, 0.01])
        self.model.axs[0].set_autoscale_on(False)

        self.model.axs[0].legend(loc='upper right')
        plt.pause(0.01)

        loss_f  = self.model.hist['loss_f']
        loss_ic = self.model.hist['loss_ic']
        loss_u  = self.model.hist['loss_u']

        loss_f.append(logs['loss_f'])
        loss_ic.append(logs['loss_ic'])
        loss_u.append(logs['loss_u'])

        self.model.axs[1].clear()
        self.model.axs[1].plot(loss_f)
        self.model.axs[1].plot(loss_ic)
        self.model.axs[1].plot(loss_u)
        self.model.axs[1].set_xlabel('Epoch')
        self.model.axs[1].set_ylabel('Error')
        self.model.axs[1].legend(['loss_f', 'loss_ic', 'loss_u'], loc='upper right')
        plt.pause(0.01)


class ODENetwork(tf.keras.Model):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    initializer = tf.keras.initializers.GlorotUniform()

    t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)
    ic_t = tf.constant(3, dtype=tf.double)
    b = tf.Variable(5, dtype=tf.double)
    k = tf.Variable(20, dtype=tf.double)

    one = tf.constant(1, dtype=tf.double)
    lam = tf.Variable(0.8, dtype=tf.double)

    loss_f_trckr  = tf.keras.metrics.Mean(name="loss_f")
    loss_ic_trckr = tf.keras.metrics.Mean(name="loss_ic")
    loss_u_trckr  = tf.keras.metrics.Mean(name="loss_u")

    freeze = False

    hist = {
        "loss_f": [],
        "loss_ic": [],
        "loss_u": [],
    }

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(top=0.90, hspace = 0.35)
    plt.show(block=False)
    plt.pause(2)

    t_train = np.arange(0, 1, 0.001)
    x_exact = np.asarray([-3.0 * t * np.exp(-7 * t) for t in t_train])

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
    def loss_f(self, f, f_t, f_tt):
        loss_f = tf.keras.backend.abs(f_tt + self.b * f_t + self.k * f)
        return loss_f

    @tf.function
    def loss_ic(self):
        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(self.t_0)
            f_0 = tf.cast(self(self.t_0, training=False), dtype=tf.double)
            f_t0 = tape_ord_1.gradient(f_0, self.t_0)
            loss_ic = tf.keras.backend.abs(f_0) + tf.keras.backend.abs(f_t0 + self.ic_t)
            loss_ic = tf.reshape(loss_ic, (1,))
        return loss_ic

    @tf.function
    def loss_u(self, u_pred, u_exact):
        loss_u = tf.keras.backend.abs(u_exact - u_pred)
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

                loss_f = tf.keras.backend.map_fn(
                    lambda x: self.loss_f(x[0], x[1], x[2]),
                    (u, f_t, f_tt),
                    dtype=tf.double)
                loss_f = tf.reduce_mean(loss_f, axis=-1)

                loss_u = tf.keras.backend.map_fn(
                    lambda x: self.loss_u(x[0], x[1]),
                    (u, x_observed),
                    dtype=tf.double)
                loss_u = tf.reduce_mean(loss_u, axis=-1)

                loss_ic = tf.reduce_mean(self.loss_ic(), axis=-1)

                loss = tf.keras.backend.abs(self.lam) * (loss_f + loss_ic) + (self.one - tf.keras.backend.abs(self.lam)) * loss_u

        self.loss_f_trckr.update_state(loss_f)
        self.loss_ic_trckr.update_state(loss_ic)
        self.loss_u_trckr.update_state(loss_u)

        grad_b = tape_ord_1.gradient(loss, self.b)
        grad_k = tape_ord_1.gradient(loss, self.k)
        self.optimizer.apply_gradients(zip([grad_b], [self.b]))
        self.optimizer.apply_gradients(zip([grad_k], [self.k]))

        grads = tape_ord_1.gradient(loss_u, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        grad_lam = tape_ord_1.gradient(loss, self.lam)
        self.optimizer.apply_gradients(zip([grad_lam], [self.lam]))

        return {
            "loss_f": self.loss_f_trckr.result(),
            "loss_ic": self.loss_ic_trckr.result(),
            "loss_u": self.loss_u_trckr.result(),
            "t": t_observed,
            "x": tf.keras.backend.squeeze(x_observed, axis=1),
            "b": self.b,
            "k": self.k,
            "lam": self.lam,
            "t_train": self.t_train,
            "x_exact": self.x_exact}

    @property
    def metrics(self):
        return [self.loss_f_trckr, self.loss_ic_trckr, self.loss_u_trckr]


def main():

    t_index, t_train       = ODENetwork.data(0, 1, 0.0001)
    x_exact                = ODENetwork.exact(t_train)
    t_observed, x_observed = ODENetwork.observed(t_index, t_train, x_exact, mu=0, sigma=0.05, points=2000)

    batch_size = 25 # len(t_train)
    epochs     = 15000 # 100000

    inputs = keras.Input(shape=(1, ))
    l1 = layers.Dense(50, activation="sigmoid")(inputs)
    l2 = layers.Dense(50, activation="sigmoid")(l1)
    l3 = layers.Dense(50, activation="sigmoid")(l2)
    outputs = layers.Dense(1, activation="linear")(l3)
    model = ODENetwork(inputs, outputs)

    print(model.summary())
    model.compile()

    t_obs_ds = tf.data.Dataset.from_tensor_slices(t_observed)
    t_obs_ds = t_obs_ds.map(lambda x: tf.cast(x, dtype=tf.double))

    x_obs_ds = tf.data.Dataset.from_tensor_slices(x_observed)
    x_obs_ds = x_obs_ds.map(lambda x: tf.cast(x, dtype=tf.double))

    diffq_ds = tf.data.Dataset.zip((t_obs_ds, x_obs_ds))
    diffq_ds = diffq_ds.shuffle(len(t_observed), seed=123)
    diffq_ds = diffq_ds.batch(batch_size)

    diffq_ds = diffq_ds.cache()

    model.fit(
        diffq_ds,
        epochs=epochs,
        workers=-1,
        verbose=0,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), MonitorCallback()])

    print("\n b: {}, k: {}".format(model.b.numpy(), model.k.numpy()))

    fig_a = plt.figure()
    ax1 = fig_a.add_subplot(111)
    ax1.scatter(t_observed, x_observed, s=10, c='red', marker="o", label='sample')
    ax1.scatter(t_train, x_exact, s=10, c='blue', marker="s", label='exact')
    ax1.scatter(t_train, model.predict(t_train), s=10, c='orange', marker="s", label='approx.')
    plt.legend(loc='lower right')
    plt.show()

    # model.save('ode2ord_paramfit.h5')

if __name__ == '__main__':
    main()