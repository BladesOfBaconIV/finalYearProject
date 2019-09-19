from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as k

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

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

model.save('../../models/dnn.h5')
