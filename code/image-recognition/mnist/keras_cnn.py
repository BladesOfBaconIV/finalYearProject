from explore_mnist import get_processed_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
import keras

from os.path import abspath

if k.image_data_format() == 'channels_first':
    INPUT_SHAPE = (1, 28, 28)
else:
    INPUT_SHAPE = (28, 28, 1)

(X_train, y_train), (X_test, y_test) = get_processed_mnist(input_shape=INPUT_SHAPE)

def make_model(num_conv=1, conv_size=64, kernel_size=(3, 3), pool_size=(2, 2),
        num_dense=1, dense_size=128, dropout=0.25):
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=INPUT_SHAPE))
    for _ in range(num_conv):
        model.add(Conv2D(conv_size, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    for _ in range(num_dense):
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=1-dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']
    )

cnn = KerasClassifier(build_fn=make_model,
    num_epochs=16,
    batch_size=128,
    verbose=0
)
gs = GridSearchCV(cnn, hyperparams, iid=True, verbose=2)
gs.fit(X_train, y_train)

best_cnn = gs.best_estimator_
print(best_cnn.score(X_test, y_test))


model.save('models/keras_cnn.h5')
