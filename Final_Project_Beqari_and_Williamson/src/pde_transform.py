import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# example:
# u_x" + u_t" + u * u_x' = sin(𝜋t) * (2 - 𝜋^2 * x^2 + 2x^3 * sin(𝜋t))
# u(x, 0) = 0
# u(x, 1) = 0
# u(0, t) = 0
# u_x'(1, t) = 2 * sin(𝜋t)


class MonitorCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        points = logs["points"]

        self.model.ax0.clear()
        self.model.fig.suptitle("Training...")
        surf0 = self.model.ax0.plot_surface(self.model.tt,
                                            self.model.xx,
                                            self.model.z,
                                            cmap=cm.winter,
                                            linewidth=0.1,
                                            antialiased=False,
                                            label='exact',
                                            facecolor='blue',
                                            edgecolor='blue')
        surf0._facecolors2d=surf0._facecolors3d
        surf0._edgecolors2d=surf0._edgecolors3d
        surf1 = self.model.ax0.plot_surface(
            self.model.tt,
            self.model.xx,
            np.asarray(self.model.psi_predict(self.model.xt_train)).reshape(
                (len(self.model.x), len(self.model.t))),
            cmap=cm.autumn,
            linewidth=0.1,
            antialiased=False,
            label='approx.',
            facecolor='orange',
            edgecolor='orange')
        surf1._facecolors2d=surf1._facecolors3d
        surf1._edgecolors2d=surf1._edgecolors3d
        scatter0 = self.model.ax0.scatter(points[:, 1],
                                          points[:, 0],
                                          -1,
                                          s=10,
                                          c='red',
                                          marker="o",
                                          label='sample')
        # self.model.ax[0].set_zlim(-0.01, 1.01)
        self.model.ax0.zaxis.set_major_locator(LinearLocator(10))
        self.model.ax0.zaxis.set_major_formatter(
            FormatStrFormatter('%.02f'))
        self.model.ax0.set_xlabel('t')
        self.model.ax0.set_ylabel('x')
        self.model.ax0.set_zlabel('pde solution value')
        self.model.ax0.legend(loc='upper right')
        plt.pause(0.01)

        loss_f  = self.model.hist['loss_f']
        loss_f.append(logs['loss_f'])

        self.model.ax1.clear()
        self.model.ax1.plot(loss_f)
        self.model.ax1.set_xlabel('Epoch')
        self.model.ax1.set_ylabel('Error')
        # self.model.axs[1].set_ylim(0.0, 0.02)
        self.model.ax1.legend(['loss_f'], loc='upper right')
        plt.pause(0.01)

class ODENetwork(tf.keras.Model):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    initializer = tf.keras.initializers.GlorotUniform()

    zero = tf.constant(0.0, dtype=tf.double)
    one  = tf.constant(1.0, dtype=tf.double)
    two  = tf.constant(2.0, dtype=tf.double)
    pi   = tf.constant(np.pi, dtype=tf.double)

    loss_f_trckr  = tf.keras.metrics.Mean(name="loss_f")
    hist = {
        "loss_f": [],
    }

    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122)
    plt.subplots_adjust(top=0.90, hspace = 0.35)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show(block=False)

    xmin =  0.0
    xmax =  1.0
    tmin =  0.0
    tmax =  1.0

    t = np.linspace(tmin, tmax, num=100)
    x = np.linspace(xmin, xmax, num=100)
    xx, tt = np.meshgrid(x, t)
    xt_train = np.column_stack((xx.ravel(), tt.ravel()))

    z = np.sin(np.pi * xt_train[:,1]) * np.power(xt_train[:,0], 2)
    z = np.reshape(z, (len(x), len(t)))

    @staticmethod
    def data(xmin, xmax, x_num, tmin, tmax, t_num):
        x = np.linspace(xmin, xmax, num=x_num)
        t = np.linspace(tmin, tmax, num=t_num)
        xx, tt = np.meshgrid(x, t)
        xt_train = np.column_stack((xx.ravel(), tt.ravel()))
        return x, t, xx, tt, xt_train

    @staticmethod
    def exact(xt_train):
        f_exact = np.sin(np.pi * xt_train[:,1]) * np.power(xt_train[:,0], 2)
        return f_exact

    @tf.function
    def psi_func(self, points, u, predict=False):

        if predict:
            x = points[0]
            t = points[1]
            points = tf.expand_dims(points, axis=0)
        else:
            x, t = tf.unstack(points, axis=1)
 
        point_1t = tf.math.multiply(points, tf.constant(np.array([0, 1]), dtype=tf.double))
        point_1t = tf.math.add(point_1t, tf.constant(np.array([1, 0]), dtype=tf.double))

        with tf.GradientTape(persistent=True) as tape_ord_1:
            tape_ord_1.watch(point_1t)
            u_1t = tf.cast(self(point_1t, training=False), dtype=tf.double)
            u1t = tape_ord_1.gradient(u_1t, point_1t)
        
        u1t_x, u1t_t = tf.unstack(u1t, axis=1)
        psi = (self.two * x *
               tf.keras.backend.sin(self.pi * t)) + (tf.keras.backend.pow(
                   t, 2) - t) * (tf.keras.backend.pow(x, 2) - x) * (u - u_1t - u1t_x)
        return psi

    @tf.function
    def loss_f(self, point, psi, dpsi_dxdt, d2psi_dx2dt2):
        x = point[0]
        t = point[1]

        dpsi_dx = dpsi_dxdt[0]
        d2psi_dx2 = d2psi_dx2dt2[0]
        d2psi_dt2 = d2psi_dx2dt2[1]

        loss_f = tf.keras.backend.abs(
            d2psi_dx2 + d2psi_dt2 + psi * dpsi_dx -
            tf.keras.backend.sin(self.pi * t) *
            (self.two - tf.keras.backend.pow(self.pi, 2) *
             tf.keras.backend.pow(x, 2) + self.two *
             tf.keras.backend.pow(x, 3) * tf.keras.backend.sin(self.pi * t)))
        return loss_f

    def train_step(self, points):

        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(points)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(points)

                u_training = tf.cast(self(points, training=True), dtype=tf.double)
                psi = self.psi_func(points, u_training)

                dpsi_dxdt = tape_ord_1.gradient(psi, points)
                d2psi_dx2dt2 = tape_ord_2.gradient(dpsi_dxdt, points)

                loss_f = tf.keras.backend.map_fn(
                    lambda x: self.loss_f(x[0], x[1], x[2], x[3]),
                    (points, psi, dpsi_dxdt, d2psi_dx2dt2),
                    dtype=tf.double)
                loss_f = tf.reduce_mean(loss_f, axis=-1)

        self.loss_f_trckr.update_state(loss_f)

        grads = tape_ord_1.gradient(loss_f, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss_f": self.loss_f_trckr.result(),
            "points": points
        }

    @property
    def metrics(self):
        return [self.loss_f_trckr]

    def psi_predict(self, points):
        u_pred = tf.cast(self.predict(points), dtype=tf.double)
        predictions = tf.keras.backend.map_fn(lambda x: self.psi_func(x[0], x[1], True), (points, u_pred), dtype=tf.double)
        return predictions




def main():

    xmin =  0.0
    xmax =  1.0
    tmin =  0.0
    tmax =  1.0

    x, t, xx, tt, xt_train = ODENetwork.data(xmin, xmax, 100, tmin, tmax, 100)
    f_exact                = ODENetwork.exact(xt_train)

    batch_size = 25 # len(xt_train)
    epochs     = 1000 # 100000

    inputs = keras.Input(shape=(2, ))
    l1 = layers.Dense(1024, activation="sigmoid")(inputs)
    l2 = layers.Dense(1024, activation="sigmoid")(l1)
    l3 = layers.Dense(1024, activation="sigmoid")(l2)
    outputs = layers.Dense(1, activation="linear")(l3)
    model = ODENetwork(inputs, outputs)

    print(model.summary())
    model.compile()

    xt_ds = tf.data.Dataset.from_tensor_slices(xt_train)
    # xt_ds = xt_ds.map(lambda x: tf.cast(x, dtype=tf.double))
    xt_ds = xt_ds.shuffle(len(xt_train), seed=123)
    xt_ds = xt_ds.batch(batch_size)
    xt_ds = xt_ds.cache()

    model.fit(xt_ds,
              epochs=epochs,
              workers=-1,
              verbose=1,
              callbacks=[
                  tf.keras.callbacks.TerminateOnNaN(),
                  MonitorCallback()
              ])

    # model.save('ode2ord_paramfit.h5')

if __name__ == '__main__':
    main()