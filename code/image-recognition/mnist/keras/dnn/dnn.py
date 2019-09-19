from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as k
import keras

from os.path import abspath

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

model = Sequential()
model.add(Dense(784, activation='relu'))
model.add(Dropout(rate=0.75, seed=42))
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.75, seed=42))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
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

filename = abspath('../../models/dnn.h5')
model.save(filename)
