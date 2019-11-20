from explore_dataset import load_data

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
import keras.backend as K

from sklearn.utils import class_weight
import numpy as np

img_shape = (256, 256)
batch_size = 32
train, val, test = load_data(
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


vgg = VGG16(include_top=False, input_shape=(*img_shape, 3), weights='imagenet')

output = vgg.layers[-1].output
output = Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

input_shape = vgg_model.output_shape[1]

model = Sequential()
model.add(vgg_model)

model.add(Dense(256, input_dim=input_shape, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=['accuracy', sensitivity, specificity]
)

print(model.summary())

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train.classes),
    train.classes
)

reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, cooldown=1)
tb = TensorBoard(log_dir='./logs/transfer_1')
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit_generator(
    train,
    steps_per_epoch=4170//batch_size,
    epochs=32,
    validation_data=val,
    validation_steps=1040//batch_size,
    verbose=1,
    class_weight=class_weights,
    callbacks=[reduce_lr, tb, early_stop]
)

model.save('transfer_1.h5')
