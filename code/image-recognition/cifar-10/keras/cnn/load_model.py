# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
from keras.datasets import cifar10
from keras import utils

# load model
model = load_model('models/basic_cnn.h5')
# summarize model.
model.summary()
# load dataset
(X_train, Y_train), (x_test, y_test) = cifar10.load_data()

y_test = utils.to_categorical(y_test, 10)
x_test = x_test.astype('float32')
x_test /= 255

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
