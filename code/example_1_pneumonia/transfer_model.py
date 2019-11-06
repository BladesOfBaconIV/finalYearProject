from explore_dataset import load_data

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

from sklearn.utils import class_weight
import numpy as np
import pickle as pkl

train, val, test = load_data(img_shape=(224, 224))

vgg = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

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
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=['accuracy']
)

print(model.summary())

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train.classes),
    train.classes
)

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit_generator(
    train,
    steps_per_epoch=4000 / 32,
    epochs=32,
    validation_data=val,
    validation_steps=1000 / 32,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stop]
)

with open('transfer_hist.pkl', 'wb') as f:
    pkl.dump(history, f)

score = model.evaluate_generator(test)
print(f'Loss: {score[0]}, Accuracy: {score[1]}')
