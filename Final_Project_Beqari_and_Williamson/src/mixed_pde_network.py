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
#   d/dx[du(x,t)/dt] = -k(x,t)u(x,t)
#   u(0,x) = 1
#   no closed solution

class ODENetwork():

    def __init__(self):
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # self.optimizer = keras.optimizers.SGD(learning_rate=0.05)
        self.history = []
    
    def get_data(self):
        x = np.linspace(-5.0, 5.0, num=1000)
        t = np.linspace(0.0, 5, num=1000)
        xx, tt = np.meshgrid(x, t)
        x_train = np.column_stack((xx.ravel(), tt.ravel()))
        return x, t, xx, tt, x_train
    
    @tf.function
    def gaussian(self, x, beta=-2):
        return tf.keras.backend.exp(-tf.keras.backend.pow(beta * x, 2))

    def create_model(self):
        inputs = keras.Input(shape=(2, ))
        l1 = layers.Dense(1024, activation=self.gaussian)(inputs)
        outputs = layers.Dense(1, activation="linear")(l1)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
        print(self.model.summary())
        return self

    @tf.function
    def k(self, x, t, lam=0.5):
        # coeff = tf.constant(lam, dtype=tf.double)
        return tf.keras.backend.exp(lam * t + x)

    @tf.function
    def loss(self, point, u, d2u_dxdt):
        point0 = tf.expand_dims(tf.math.multiply(
            tf.cast(point, dtype=tf.double), 
            tf.constant(np.array([1, 0]), dtype=tf.double)), 
            axis=0)
        u_x0  = tf.cast(self.model(point0, training=False), dtype=tf.double)  
        x = tf.cast(point[0], dtype=tf.double)
        t = tf.cast(point[1], dtype=tf.double)
        loss = tf.keras.backend.abs(u_x0 - 1) + tf.keras.backend.abs(d2u_dxdt + u * self.k(x, t))
        return loss
    
    @tf.function
    def apply_training_step(self, point):

        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(point)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(point)
                
                # function value
                u = tf.cast(self.model(point, training=True), dtype=tf.double)
                
                # first derivatives
                grads = tape_ord_1.gradient(u, point)
                du_dx = grads[:, 0]
                # du_dt = grads[:, 1]

                # second and mixed derivatives
                grads2x = tape_ord_2.gradient(du_dx, point)
                # d2u_dx2  = grads2x[:, 0]
                d2u_dxdt = grads2x[:, 1]
                # grads2t = tape_ord_2.gradient(du_dt, point)
                # d2u_dtdx = grads2t[:, 0]
                # d2u_dt2  = grads2t[:, 1]

                # print("check: {} == {} ".format(d2u_dxdt,  d2u_dtdx))

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss(x[0], x[1], x[2]), 
                    (point, u, d2u_dxdt), 
                    dtype=tf.double)
                loss = tf.reduce_mean(loss, axis=-1)
                   
        grads = tape_ord_1.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def train(self):
        epochs = 3
        batch_size = len(self.get_data())
        # t_0  = tf.constant(np.array([[0.0]]), dtype=tf.double)

        x, t, xx, tt, x_train = self.get_data()
        xt_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        xt_dataset = xt_dataset.shuffle(buffer_size=1024, seed=1234).batch(batch_size)

        for epoch in range(epochs):            
  
            for step, point in enumerate(xt_dataset):
                loss = self.apply_training_step(point)

                if step % 200 == 0:
                    self.history.append(tf.reduce_mean(loss))
                    print("Training loss for step/epoch {}/{}: {}".format(step, epoch, tf.reduce_mean(loss)))

    def predict(self):
        x, t, xx, tt, x_train = self.get_data()
        predictions = self.model.predict(x_train)
        return predictions

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    x, t, xx, tt, x_train = ode_net.get_data()
    z = ode_net.predict().reshape((t.shape[0], x.shape[0]))
    # Plot the surface.
    surf = ax.plot_surface(tt, xx, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    plt.title('ODE Solution')
    plt.ylabel('x')
    plt.xlabel('t')
    plt.contourf(t, x, z)
    plt.show()

if __name__ == '__main__':
    main()
