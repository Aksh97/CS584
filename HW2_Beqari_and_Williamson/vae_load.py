import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

def print_greyscale(pixels_1, pixels_2, width=28, height=28):
    
    def get_single_greyscale(pixel):
        val = 232 + np.round(pixel * 23)
        return '\x1b[48;5;{}m \x1b[0m'.format(int(val))

    for l in range(height):
        line_pixels = np.concatenate((pixels_1[l * width:(l+1) * width], pixels_2[l * width:(l+1) * width]), axis=None)
        print(''.join(get_single_greyscale(p) for p in line_pixels))

vae = load_model('vae_model')

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()


x_test = np.load("MNISTXtrain1_noisy_2.5.npy")

test_input = np.expand_dims(x_test[1], -1).astype("float32") / 255
test_input = np.reshape(test_input, (1, 28, 28, 1))

test_img = np.resize(x_test[1], (756,))
result = vae.predict(test_input)
result = np.resize(result, (756,))
print_greyscale(test_img, result)