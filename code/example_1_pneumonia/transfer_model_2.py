from explore_dataset import load_data

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import keras.backend as K

from sklearn.utils import class_weight
from collections import Counter
import numpy as np
import pickle as pkl

img_shape = (256, 256)
batch_size = 32
train, val, test = load_data(
    img_shape=img_shape,
    batch_size=batch_size,
    width_shift_range=5,
    height_shift_range=5,
    horizontal_flip=True,
    cross_val=0.2
)

vgg = VGG16(include_top=False, input_shape=(*img_shape, 3), weights='imagenet')

output = vgg.layers[-1].output
output = Flatten()(output)
output = Dense(1, activation='sigmoid')(output)

vgg_model = Model(inputs=vgg.input, outputs=output)

for layer in vgg_model.layers[:18]:
    layer.trainable = False

for layer in vgg_model.layers:
    print(f'layer: {layer.name}, trainable: {layer.trainable}')

vgg_model.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=['accuracy']
)

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train.classes),
    train.classes
)

# counter = Counter(train.classes)
# t = 0.4  # do we focus on sensitivity (t = 1), or specificity (t = 0)
# class_weights = {1: (counter[0]/counter[1]) * t}

reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, cooldown=1)
tb = TensorBoard(log_dir='./logs/transfer_2_1')
checkpoint = ModelCheckpoint(
    filepath='transfer_2_1_{epoch:02d}.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

history = vgg_model.fit_generator(
    train,
    steps_per_epoch=4170//batch_size,
    epochs=32,
    validation_data=val,
    validation_steps=1040//batch_size,
    verbose=1,
    class_weight=class_weights,
    callbacks=[reduce_lr, tb, checkpoint]
)

with open('transfer_2_1.pkl', 'wb') as f:
    pkl.dump(history, f)
