from explore_dataset import load_data

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy
import keras.backend as K

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import pickle as pkl

from itertools import cycle

img_shape = (256, 256)
batch_size = 16
train, val, test = load_data(img_shape=img_shape, batch_size=batch_size)
train_aug, *_ = load_data(
    img_shape=img_shape,
    batch_size=batch_size,
    width_shift_range=10,
    height_shift_range=5,
    horizontal_flip=True
)

class_weights = compute_class_weight(
    'balanced',
    np.unique(train.classes),
    train.classes
)

input_shape = (3, *img_shape) if K.image_data_format() == 'channels_first' else (*img_shape, 3)


def make_model(optimizer):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss=binary_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def fit_model(m, t):
    return m.fit_generator(
        t,
        steps_per_epoch=2600/batch_size,
        epochs=50,
        validation_data=val,
        validation_steps=1,
        class_weight=class_weights
    )


models = [make_model(opt) for opt in (Adam(), Adam(), RMSprop(), RMSprop())]
hists = [fit_model(m, t) for m, t in zip(models, cycle((train, train_aug)))]
filenames = ['pure_adam', 'aug_adam', 'pure_rms', 'aug_rms']

for filename, hist in zip(filenames, hists):
    with open(f'{filename}.pkl', 'wb') as f:
        pkl.dump(hists, f)