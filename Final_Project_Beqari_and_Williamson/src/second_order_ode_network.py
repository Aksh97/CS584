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
        l1 = layers.Dense(1024, activation=self.gaussian)(inputs)
        outputs = layers.Dense(1, activation="linear")(l1)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="experimental")
        print(self.model.summary())
        return self

    @tf.function
    def loss(self, t, u, u_t, u_tt, t_0):
        b    = tf.constant(14, dtype=tf.double)
        k    = tf.constant(49, dtype=tf.double)
        bc_t = tf.constant(3, dtype=tf.double)
        with tf.GradientTape(persistent=True) as tape_ord_1_bc:
            tape_ord_1_bc.watch(t_0)
            u_0  = tf.cast(self.model(t_0, training=False), dtype=tf.double)
            u_t0 = tape_ord_1_bc.gradient(u_0, t_0)
            loss = tf.keras.backend.abs(u_0) + tf.keras.backend.abs(u_t0 + bc_t) + tf.keras.backend.abs(u_tt + b * u_t + k * u)
            # loss = tf.keras.backend.square(u_0) + tf.keras.backend.square(u_t0 + bc_t) + tf.keras.backend.square(u_tt + b * u_t + k * u)
        return loss
    
    @tf.function
    def apply_training_step(self, t, t_0):
        with tf.GradientTape(persistent=True) as tape_ord_2:
            tape_ord_2.watch(t)

            with tf.GradientTape(persistent=True) as tape_ord_1:
                tape_ord_1.watch(t)
                u_training = self.model(t, training=True)
                u = tf.cast(self.model(t, training=False), dtype=tf.double)
                u_t = tape_ord_1.gradient(u_training, t)
                u_tt = tape_ord_2.gradient(u_t, t)

                loss = tf.keras.backend.map_fn(
                    lambda x: self.loss(x[0], x[1], x[2], x[3], t_0), 
                    (t, u, u_t, u_tt), 
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
        predictions = self.model.predict(self.get_data())
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

    plt.plot(ode_net.exact_solution())
    plt.plot(ode_net.predict())
    plt.title('ODE Solution')
    plt.ylabel('x')
    plt.xlabel('step')
    plt.legend(['exact', 'approx.'], loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
