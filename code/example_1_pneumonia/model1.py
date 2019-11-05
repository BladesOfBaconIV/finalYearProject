from explore_dataset import load_data
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import binary_crossentropy

img_shape = (256, 256)
train_gen, test_gen = load_data()


model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    input_shape=(*img_shape, 3),
    activation='relu'
))
model.add(Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss=binary_crossentropy, optimizer=Adam(),metrics=['accuracy'])

model.fit_generator(train_gen, epochs=8, steps_per_epoch=5400/32, class_weight={0: 3., 1: 1.})

try:
    model.save('m1.h5')
except Exception:
    pass

score = model.evaluate_generator(train_gen, steps=600/32)
print(f'Loss: {score[0]}, Accuracy {score[1]}')
