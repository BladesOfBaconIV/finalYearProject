from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import Counter
from itertools import starmap
import numpy as np

def get_processed_mnist(input_shape=()):
    """returns the mnist dataset with X normalised, and y categorised
    input_shape: if provided will reshape the images in X,
                 e.g. from (28, 28) -> (784)"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # normalises pixel values to between 0 and 1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    # converts y from an int to an array index
    # i.e 7 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    if input_shape:
        # reshapes the images in X_train and X_test
        num_train, *_ = X_train.shape
        num_test, *_ = X_test.shape
        X_train = X_train.reshape(num_train, *input_shape)
        X_test = X_test.reshape(num_test, *input_shape)
    return (X_train, y_train), (X_test, y_test)

def visualise_mnist():
    """Provides a visual representation of the data in the mnist dataset"""
    # barcharts showing how many of each number in the training and test data
    bar_fig, bar_axes = plt.subplots(1, 2, sharey=True)
    for i, (ax, labels) in enumerate(zip(bar_axes, map(Counter, (y_train, y_test)))):
        bars = ax.bar(sorted(map(str, labels.keys())), labels.values())
        [ax.text(b.get_x(), b.get_height()+10, f'{b.get_height()}') for b in bars]
        ax.set_xlabel('Number')
        ax.set_ylabel('Count')
        ax.set_title('Test Data' if i else 'Training Data')

    # Shows the first image of each digit from the training data
    im_fig, im_axes = plt.subplots(2, 5)
    y_train_list = list(y_train)
    for n, ax in enumerate(im_axes.flatten()):
        image = X_train[y_train_list.index(n)]
        ax.imshow(image, cmap='gray_r')
        ax.set_axis_off()
    im_fig.suptitle('Example images')

    plt.show()

if __name__=="__main__":
    visualise_mnist()
