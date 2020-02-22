import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict


class BfgsTrainingMonitor:

    def __init__(self, model, data, metrics):
        """
        Object to monitor the training metrics of a BfgsMlp
        Records the metrics and loss of the model before each call of the val_and_grad_func by minimize_bfgs
            in a defaultdict with keys of '{dataset_name}_{metric}', e.g. 'training_loss'
        :param model: BfgsMlp model to be monitored, must be compiled before fitting
        :param data: data to evaluate the model on,
                should be a dict of the form {'dataset_name': (features, labels), ..}
        :param metrics: list of metrics the model was compiled with
        """
        self.history = defaultdict(list)
        self.data = data
        self.metrics = metrics
        self.model = model

    def monitor(self, func):
        """
        A wrapper function for BfgsMlp val_and_grad_func
        Records the loss and model metrics before the function is called
        :param func: function to be wrapped
        :return:
        """
        def wrapper(*args, **kwargs):
            for dataset, (X, y) in self.data.items():
                loss, *results = self.model.evaluate(X, y, verbose=0)
                self.history[f'{dataset}_loss'].append(loss)
                assert len(results) == len(self.metrics)
                for result, metric in zip(results, self.metrics):
                    self.history[f'{dataset}_{metric}'].append(result)
            return func(*args, **kwargs)
        return wrapper


class BfgsMlp(tf.keras.Sequential):

    def __init__(self, n_input, n_hidden, n_output, loss=tf.keras.losses.MeanSquaredError()):
        """
        A model for a Multi-layered perceptron trained using the BFGS optimizer algorithm from TensorFlow Probability
        Creates a MLP with a single hidden layer
        Extends tf.keras.Sequential, and overrides the fit method
        :param n_input: number of neurons in the input layer
        :param n_hidden: number of neurons in the hidden layer
        :param n_output: number of neurons in the ouptut layer
        :param loss: loss function, default is MSE, should be an instance of tf.keras.losses.Loss
        """
        super().__init__([
            tf.keras.Input(shape=(n_input,)),
            tf.keras.layers.Dense(n_hidden, 'relu'),
            tf.keras.layers.Dense(n_output, 'softmax'),
        ])
        # Needed to train the model using a 1D tensor of params
        self._trainable_shapes, self._indexes_1d, self._partitions_1d = self._make_1d_mapping()
        self.loss = loss

    def fit(self, X, y, optimizer=tfp.optimizer.bfgs_minimize, monitor=None, **kwargs):
        """
        Fit the model to some data. Overrides tf.keras.Sequential fit()
        :param X: Features to train on
        :param y: Labels to compare to, should be categorical not sparse, e.g. [0, 0, 1, 0] not 2
        :param optimizer: The optimizer function to calculate weight updates,
            default tfp.optimizer.bfgs_minimize, should work any optimizer that needs a value_and_gradients function
        :param monitor: Instance of BfgsTrainingMonitor to record training history
        :param kwargs: extra arguments to pass to the optimizer function
        :return: None
        """
        val_and_grad_func = self._make_val_and_grad_func(X, y)
        if monitor:
            val_and_grad_func = monitor.monitor(val_and_grad_func)
        result = optimizer(
            val_and_grad_func,  # values_and_grad_func
            self._get_weights_1d(),     # initial_position
            **kwargs
        )
        self._update_weights(result.position)  # Needs one final update with the final position

    def accuracy(self, X, y):
        """
        Accuracy of the model
        :param X: Data to predict
        :param y: True labels
        :return: Accuracy of model (y_hat == y_true
        """
        _y = np.argmax(y, axis=1)
        _y_hat = np.argmax(self.predict(X), axis=1)
        return sum(_y == _y_hat) / len(_y)

    def _make_val_and_grad_func(self, X, y):
        """
        Factory function to make val_and_grad_func for the optimizer
        :param X: data to be trained on
        :param y: labels for X
        :return: val_and_grad_func
        """
        @tf.function
        def value_and_grad(weights_1d):
            """
            Function to return the loss and gradient of the error surface w.r.t the model weights
            :param weights_1d: 1D tensor of new weights for the model calculated by the optimizer function
            :return: loss and gradient value
            """
            # used to watch changes to the model to calculate the gradient w.r.t the variable
            with tf.GradientTape() as tape:
                self._update_weights(weights_1d)
                loss_value = self.loss(self(X, training=True), y)

            # calculate gradient w.r.t trainable_values (i.e weights) and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.trainable_variables)
            grads = tf.dynamic_stitch(self._indexes_1d, grads)
            return loss_value, grads
        return value_and_grad

    def _update_weights(self, weights_1d):
        # Update the weights using a 1D tensor
        # Split the 1D tensor into n 1D tensors, 2 for each weight layers (weights and biases)
        weights_each_layer = tf.dynamic_partition(weights_1d, self._partitions_1d, len(self._indexes_1d))
        for layer, weights in zip(self.trainable_variables, weights_each_layer):
            layer.assign(tf.reshape(weights, layer.shape))

    def _get_weights_1d(self):
        return tf.dynamic_stitch(self._indexes_1d, self.trainable_variables)

    def _make_1d_mapping(self):
        """
        Function to make the mapping to convert to and from a 1D weight tensor back to the original weight shapes
        :return: trainable_shapes, indexes, partitions
        """
        trainable_shapes = tf.shape_n(self.trainable_variables)

        indexes = []  # indexes to map from original shapes to 1D tensor
        partitions = []  # which layer each param in 1D tensor should be mapped to

        n = 0  # offset for current layer indexes
        for layer, shape in enumerate(trainable_shapes):
            num_values = np.product(shape)  # number of trainable params in layer
            # create an index for current layer in current layer shape
            indexes.append(tf.reshape(tf.range(n, n + num_values), shape))
            # Update the list of partitions
            partitions.extend([layer] * num_values)
            n += num_values

        return trainable_shapes, indexes, partitions
