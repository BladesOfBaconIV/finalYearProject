from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath

DATA_DIR = abspath('../../datasets/chest-xray-pneumonia/chest_xray')
CLASSES = ['NORMAL', 'PNEUMONIA']


def load_data(batch_size=32) -> tuple:
    """Returns a pair of (Train, Test) generators for the data

    :param batch_size: Number of images returned by train_gen each iteration
    :return: (train_gen, test_gen)"""

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        validation_split=0.2
    ).flow_from_directory(
        f'{DATA_DIR}/train',
        classes=CLASSES,
        class_mode='binary',
        batch_size=batch_size
    )

    test_gen = ImageDataGenerator(
        rescale=1. / 255
    ).flow_from_directory(
        f'{DATA_DIR}/test',
        classes=CLASSES,
        class_mode='binary',
        batch_size=batch_size
    )
    return train_gen, test_gen


def visualise() -> None:
    train, _ = load_data()
    train_x, train_y = next(train)
    p = np.where(train_y == 1)[0]
    n = np.where(train_y == 0)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.imshow(train_x[n[0]])
    ax1.axis('off')
    ax1.set_title('Normal')

    ax2.imshow(train_x[p[0]])
    ax2.axis('off')
    ax2.set_title('Pneumonia')

    plt.show()


if __name__ == "__main__":
    visualise()
