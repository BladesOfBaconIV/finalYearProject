import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from explore_dataset import load_preproccessed_dataset
from bfgs_objects import BfgsMlp, BfgsTrainingMonitor

from collections import defaultdict

# use float64 by default
tf.keras.backend.set_floatx("float64")

# prepare training data
(X, y), _ = load_preproccessed_dataset(test_split=0.0, include_grades=True)
num_classes = 5
num_features = X.shape[1]

standardise = StandardScaler()
X = standardise.fit_transform(X)

# Labels need to be 0-4, not 4-5 for categorical
y -= 1
y = tf.keras.utils.to_categorical(y).astype(np.float64)

# Tuning H
accuracies = defaultdict(list)
for H in range(0, 10, 2):  # check H in {0, 2, 4, 6, 8}
    for i in range(10):  # 10-fold cross-validation
        print(f'H: {H}, Fold {i+1}')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

        model = BfgsMlp(n_input=num_features, n_hidden=H, n_output=num_classes)
        model.fit(train_x, train_y, max_iterations=100)
        test_acc = model.accuracy(test_x, test_y)

        accuracies[H].append(test_acc)

accuracies = {H: sum(accs)/len(accs) for H, accs in accuracies.items()}
for H, acc in accuracies.items():
    print(f'Average accuracy for H = {H}: {acc*100:.2f}%')

# Once best H value is found make a model with this H and monitor training to check for over-fitting
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
best_H = max(accuracies, key=accuracies.get)
metrics = ['accuracy']

model_opt_H = BfgsMlp(num_features, best_H, num_classes)
model_opt_H.compile(  # model needs to be compiled for monitor to use evaluate
    loss=model_opt_H.loss,
    metrics=metrics
)

data = {
    'training': (train_x, train_y),
    'validation': (test_x, test_y),
}
monitor = BfgsTrainingMonitor(model_opt_H, data, metrics)
model_opt_H.fit(train_x, train_y, monitor=monitor, max_iterations=100)

plt.figure(1)
plt.plot(monitor.history['training_loss'])
plt.plot(monitor.history['validation_loss'])
plt.legend(["Training loss", "Validation loss"])
plt.title("Training vs. Validation Loss")
plt.xlabel("No. calls to val_and_grad_func")
plt.ylabel("Loss")

plt.figure(2)
plt.plot(monitor.history['training_accuracy'])
plt.plot(monitor.history['validation_accuracy'])
plt.legend(["Training accuracy", "Validation Accuracy"])
plt.title("Training vs. Validation Accuracy")
plt.xlabel("No. calls to val_and_grad_func")
plt.ylabel("Accuracy")

plt.show()

# Once best number of epochs found train models with best H and epochs
min_loss = np.argmin(monitor.history["validation_loss"])//3
print(f'Min loss at {min_loss} epochs')

# Final models
models = []
for i in range(5):
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1)

    model = BfgsMlp(num_features, best_H, num_classes)
    model.fit(train_x, train_y, max_iterations=min_loss.astype('int32'))
    test_acc = model.accuracy(val_x, val_y)

    models.append((model, test_acc))

# test best model on maths data
best_model, acc = max(models, key=lambda x: x[1])

(maths_x, maths_y), _ = load_preproccessed_dataset(test_split=0.0, subject='mat', include_grades=True)
# Use the StandardScaler fitted on the Portuguese data
maths_x = standardise.transform(maths_x)

# Labels need to be 0-4, not 4-5 for categorical
maths_y -= 1
maths_y = tf.keras.utils.to_categorical(maths_y).astype(np.float64)

maths_acc = best_model.accuracy(maths_x, maths_y)

print(f'Accuracy on maths data: {maths_acc*100:.2f}%')