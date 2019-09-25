from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

_, (X, y) = mnist.load_data()

X = X.reshape(10000, 28, 28, 1)
X = X.astype('float32')
X /= 255.

model = load_model('../models/cnn.h5')

predictions = model.predict(X, batch_size=128)
predictions = predictions.argmax(axis=-1)

cf = confusion_matrix(y, predictions)
sns.heatmap(cf, annot=True, fmt='d')
plt.show()
