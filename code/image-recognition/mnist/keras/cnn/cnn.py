from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
import keras

from os.path import abspath

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train, *img_shape = X_train.shape
num_test, *_ = X_test.shape
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(num_train, 1, *img_shape)
    X_test = X_test.reshape(num_test, 1, *img_shape)
    input_shape = (1, *img_shape)
else:
    X_train= X_train.reshape(num_train, *img_shape, 1)
    X_test = X_test.reshape(num_test, *img_shape, 1)
    input_shape = (*img_shape, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.75))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy']
)

model.fit(X_train, y_train,
          batch_size=128,
          epochs=16,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tbCallBack]
)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

filename = abspath('../../models/cnn.h5')
model.save(filename)
