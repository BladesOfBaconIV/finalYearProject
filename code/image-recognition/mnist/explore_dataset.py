from keras.datasets import mnist
import matplotlib.pyplot as plt
from collections import Counter
from itertools import starmap
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

bar_fig, bar_axes = plt.subplots(1, 2, sharey=True)
for i, (ax, labels) in enumerate(zip(bar_axes, map(Counter, (y_train, y_test)))):
    bars = ax.bar(sorted(map(str, labels.keys())), labels.values())
    [ax.text(b.get_x(), b.get_height()+10, f'{b.get_height()}') for b in bars]
    ax.set_xlabel('Number')
    ax.set_ylabel('Count')
    ax.set_title('Test Data' if i else 'Training Data')

im_fig, im_axes = plt.subplots(2, 5)
y_train_list = list(y_train)
for n, ax in enumerate(im_axes.flatten()):
    image = X_train[y_train_list.index(n)]
    ax.imshow(image, cmap='gray_r')
    ax.set_axis_off()
im_fig.suptitle('Example images')

plt.show()
