from pyo import *
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt


def get_mnist_data():


    # Model / data parameters
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # fig = plt.figure
    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()
    # print(y_train[0])

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #make training set smaller for faster training
    x_train = reduce_data(x_train)
    y_train = reduce_data(y_train)

    # print(x_train.shape)
    # print(y_train.shape)
    # fig = plt.figure
    # plt.imshow(x_train[0].reshape(28,28), cmap='gray')
    # plt.show()
    # print(y_train[0])

    return x_train, y_train, x_test, y_test

def reduce_data(data, fraction=0.1, seed=42):

    """Return fraction of data"""

    random.seed(seed)
    rand_idx = random.sample(range(0,data.shape[0]), int(data.shape[0]*fraction))
    frac_data = data[rand_idx]

    return frac_data


def get_mnist_model():

    input_shape = (28,28,1)
    num_classes = 10

    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )
    # model.summary()

    return model
