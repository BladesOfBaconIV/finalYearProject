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
train, val, test = load_data(img_shape=img_shape, batch_size=batch_size, cross_val=0.2)
train_aug, val_aug = load_data(
    img_shape=img_shape,
    batch_size=batch_size,
    width_shift_range=10,
    height_shift_range=5,
    horizontal_flip=True,
    cross_val=0.2
)


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


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
        metrics=[sensitivity, specificity]
    )
    return model


def fit_model(m, t, v):
    return m.fit_generator(
        t,
        steps_per_epoch=2600/batch_size,
        epochs=24,
        validation_data=v,
        validation_steps=1,
        class_weight=class_weights
    )


models = [make_model(opt) for opt in (Adam(), Adam(), RMSprop(), RMSprop())]
hists = [fit_model(m, t) for m, (t, v) in zip(models, cycle(((train, val), (train_aug, val_aug))))]
model_names = ['pure_adam', 'aug_adam', 'pure_rms', 'aug_rms']

with open('cnn_training.pkl', 'wb') as f:
    pkl.dump(dict(zip(model_names, hists)), f)