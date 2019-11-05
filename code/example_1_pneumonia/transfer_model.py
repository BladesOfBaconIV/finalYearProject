from explore_dataset import load_data

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

import pandas as pd

train, val, test = load_data(img_shape=(224, 224))

vgg = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

output = vgg.layers[-1].output
output = Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
layer_info = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

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

model.fit_generator(
    train,
    steps_per_epoch=4000/32,
    epochs=16,
    validation_data=val,
    verbose=1
)

model.save('example_transfer.h5')

score = model.evaluate_generator(test)
print(f'Loss: {score[0]}, Accuracy: {score[1]}')