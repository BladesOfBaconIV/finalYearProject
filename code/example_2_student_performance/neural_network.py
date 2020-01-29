import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from explore_dataset import load_preproccessed_dataset
from collections import defaultdict

class MLP(tf.keras.Sequential):

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__([
            tf.keras.Input(shape=(n_input,)),
            tf.keras.layers.Dense(n_hidden, 'relu'),
            tf.keras.layers.Dense(n_output, 'softmax'),
        ])

    def make_value_grad_func(self, loss, train_x, train_y):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.
        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.trainable_variables[i].assign(tf.reshape(param, shape))

        @tf.function
        def value_and_grad_function(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.
            This function is created by function_factory.
            Args:
               params_1d [in]: a 1D tf.Tensor.
            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = loss(self(train_x, training=True), train_y)

                # calculate gradients and convert to 1D tf.Tensor
                grads = tape.gradient(loss_value, self.trainable_variables)
                grads = tf.dynamic_stitch(idx, grads)

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        value_and_grad_function.iter = tf.Variable(0)
        value_and_grad_function.idx = idx
        value_and_grad_function.part = part
        value_and_grad_function.shapes = shapes
        value_and_grad_function.assign_new_model_parameters = assign_new_model_parameters

        return value_and_grad_function


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

    # For H in {0, 2, 4, 6, 8}
    for H in range(0, 10, 2):
        for i in range(10):
            print(f'H: {H}, Fold {i+1}')
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

            pred_model = MLP(n_input=X.shape[1], n_hidden=H, n_output=num_classes)

            val_and_grad_func = pred_model.make_value_grad_func(tf.keras.losses.MeanSquaredError(), train_x, train_y)

            # convert initial model parameters to a 1D tf.Tensor
            init_params = tf.dynamic_stitch(val_and_grad_func.idx, pred_model.trainable_variables)

            # train the model with BFGS solver
            results = tfp.optimizer.bfgs_minimize(
                value_and_gradients_function=val_and_grad_func, initial_position=init_params, max_iterations=100)

            # after training, the final optimized parameters are still in results.position
            # so we have to manually put them back to the model
            val_and_grad_func.assign_new_model_parameters(results.position)

            test_y = np.argmax(test_y, axis=1)
            test_y_hat = np.argmax(pred_model.predict(test_x), axis=1)
            test_acc = sum(test_y == test_y_hat)/len(test_y)

            accuracies[H].append(test_acc)

for H, accs in accuracies.items():
    print(f'Average accuracy for H = {H}: {sum(accs)/len(accs)}')