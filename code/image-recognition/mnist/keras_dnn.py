from explore_mnist import get_processed_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras.backend as k
import keras

from os.path import abspath

(X_train, y_train), (X_test, y_test) = get_processed_mnist(input_shape=(784,))
def make_model(num_layers=1, layer_size=256, dropout=0.25):

    model = Sequential()
    model.add(Dense(784, activation='relu'))
    for _ in range(num_layers):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']
    )

    return model

hyperparams = {
    'num_layers': [1, 5, 10],
    'layer_size': [128, 256, 512],
    'dropout': [0.2, 0.25, 0.5],
}

dnn = KerasClassifier(build_fn=make_model,
    epochs=12,
    batch_size=128,
    verbose=0
)
gs = GridSearchCV(dnn, hyperparams, iid=True, verbose=2)
gs.fit(X_train, y_train)

best_dnn = gs.best_estimator_
print(best_dnn.evaluate(X_test, y_test))

model.save('models/dnn.h5')
