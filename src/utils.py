from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np


def convert_x(np_x):
    # Normalize: / 255
    # [None, 28, 28]
    tensor = tf.convert_to_tensor(np_x/255, dtype=tf.float32)

    # [None, 28, 28, 1]
    tensor = tf.expand_dims(tensor, -1)

    return tensor


def convert_y(np_y):
    # y must be float32
    tensor = tf.convert_to_tensor(np_y, dtype=tf.float32)

    return tensor


def load_preprocess_data():
    # Put 'mnist.npz' in your ~/.keras/datasets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

    x_train = convert_x(x_train)
    x_test = convert_x(x_test)

    y_train = convert_y(y_train)
    y_test = convert_y(y_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_preprocess_data()

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
