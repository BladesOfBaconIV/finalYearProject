import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from itertools import product

# input image dimensions
img_rows, img_cols = 32, 32
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_sizes = [100, 200]
num_epochs = [64, 128]
num_conv_layers = [2, 3,]
conv_sizes = [32, 64,]

training_info = {}
for batch_size, epochs, num_conv, conv_size in product(batch_sizes, num_epochs, num_conv_layers, conv_sizes):
    model_name = f'cnn_{batch_size}_{epochs}_{num_conv}_{conv_size}'
    tbCallBack = keras.callbacks.TensorBoard(log_dir=f'/logs/rough-optimisation/{model_name}', histogram_freq=0)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))

    for _ in range(num_conv):
        model.add(Conv2D(conv_size, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack])
    score = model.evaluate(x_test, y_test, verbose=0)
    training_info[model_name] = f'Test loss: {score[0]}, Test accuracy: {score[1]}'

with open('training_overview', 'w+') as f:
    for model, info in training_info.items():
        f.write(f'{model}; {info}')
