from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from os.path import abspath

DATA_DIR = abspath('../../datasets/chest-xray-pneumonia/chest_xray')
CLASSES = ['NORMAL', 'PNEUMONIA']


def load_data(batch_size=32, img_shape=(256, 256), cross_val=None, **extra_augmentation) -> tuple:
    """Returns a tuple of (Train, Val, Test) generators for the data

    :param cross_val: Percent of training set to use as cross validation, if None uses 16 validation images
    :param batch_size: Number of images returned by train_gen each iteration
    :param img_shape: shape of the image to be returned (excluding channels)
    :param **extra_augmentation: Any extra augmentation arguments for training ImageDataGenerator
    :return: (train, val, test)"""

    val_split = 0.0 if cross_val is None else cross_val

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_split,
        **extra_augmentation
    )
    test_gen = ImageDataGenerator(
        rescale=1. / 255
    )
    flow_args = {
        'classes': CLASSES,
        'class_mode': 'binary',
        'batch_size': batch_size,
        'target_size': img_shape,
    }

    train = train_gen.flow_from_directory(f'{DATA_DIR}/train', subset='training', **flow_args)
    test = test_gen.flow_from_directory(f'{DATA_DIR}/test', **flow_args)
    if cross_val is not None:
        val = train_gen.flow_from_directory(f'{DATA_DIR}/train', subset='validation', **flow_args)
    else:
        flow_args['batch_size'] = 16
        val = test_gen.flow_from_directory(f'{DATA_DIR}/val', **flow_args)

    return train, val, test


def visualise() -> None:
    train, *_ = load_data()
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
