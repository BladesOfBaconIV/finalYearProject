import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from explore_dataset import load_preproccessed_dataset
from collections import defaultdict


class BfgsTrainingMonitor:

    def __init__(self):
        self.history = defaultdict(list)

    def monitor_loss(self, func):
        def wrapper(*args, **kwargs):
            loss, grads = func(*args, **kwargs)
            self.history['loss'].append(loss)
            return loss, grads
        return wrapper


class BfgsMlp(tf.keras.Sequential):

    def __init__(self, n_input, n_hidden, n_output, loss=tf.keras.losses.MeanSquaredError()):
        super().__init__([
            tf.keras.Input(shape=(n_input,)),
            tf.keras.layers.Dense(n_hidden, 'relu'),
            tf.keras.layers.Dense(n_output, 'softmax'),
        ])
        self._trainable_shapes, self._indexes_1d, self._partitions_1d = self._make_1d_mapping()
        self.loss = loss

    def fit(self, X, y, optimizer=tfp.optimizer.bfgs_minimize, **kwargs):
        monitor = BfgsTrainingMonitor()
        val_and_grad_func = self._make_vals_and_grad_func(X, y)
        result = optimizer(
            val_and_grad_func,  # values_and_grad_func
            self._get_weights_1d(),     # initial_position
            **kwargs
        )
        self._update_weights(result.position)
        return monitor

    def _make_vals_and_grad_func(self, X, y):
        @tf.function
        def values_and_grads(weights_1d):
            with tf.GradientTape() as tape:
                self._update_weights(weights_1d)
                loss_value = self.loss(self(X, training=True), y)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.trainable_variables)
            grads = tf.dynamic_stitch(self._indexes_1d, grads)
            return loss_value, grads
        return values_and_grads

    def _update_weights(self, weights_1d):
        weights_each_layer = tf.dynamic_partition(weights_1d, self._partitions_1d, len(self._indexes_1d))
        for layer, (weights, shape) in enumerate(zip(weights_each_layer, self._trainable_shapes)):
            self.trainable_variables[layer].assign(tf.reshape(weights, shape))

    def _get_weights_1d(self):
        return tf.dynamic_stitch(self._indexes_1d, self.trainable_variables)

    def _make_1d_mapping(self):
        trainable_shapes = tf.shape_n(self.trainable_variables)

        indexes = []  # indexes to map from 1D tensor to original shapes
        partitions = []  # which layer each param in 1D tensor should be mapped to

        i = 0
        for layer, shape in enumerate(trainable_shapes):
            num_values = np.product(shape)  # number of trainable params in layer
            indexes.append(tf.reshape(tf.range(i, i + num_values), shape))
            partitions.extend([layer] * num_values)
            i += num_values

        return trainable_shapes, indexes, partitions


if __name__ == "__main__":
    # use float64 by default
    tf.keras.backend.set_floatx("float64")

    # prepare training data
    (X, y), _ = load_preproccessed_dataset(test_split=0.0, include_grades=True)
    num_classes = 5

    standardise = StandardScaler()
    X = standardise.fit_transform(X)

    y -= 1
    y = tf.keras.utils.to_categorical(y).astype(np.float64)

    accuracies = defaultdict(list)

    m = BfgsMlp(32, 4, 5)
    hist = m.fit(X, y, max_iterations=100)
    plt.plot(hist.history['loss'])
    plt.show()

    # for H in range(0, 10, 2):
    #     for i in range(10):
    #         print(f'H: {H}, Fold {i+1}')
    #         train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
    #
    #         model = BfgsMlp(n_input=X.shape[1], n_hidden=H, n_output=num_classes)
    #
    #         model.fit(train_x, train_y, max_iterations=100)
    #
    #         test_y = np.argmax(test_y, axis=1)
    #         test_y_hat = np.argmax(model.predict(test_x), axis=1)
    #         test_acc = sum(test_y == test_y_hat)/len(test_y)
    #
    #         accuracies[H].append(test_acc)

for H, accs in accuracies.items():
    print(f'Average accuracy for H = {H}: {sum(accs)/len(accs)}')