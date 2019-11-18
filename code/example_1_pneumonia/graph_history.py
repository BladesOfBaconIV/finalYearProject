import pickle as pk
import matplotlib.pyplot as plt

with open('cnn_training.pkl', 'rb') as f:
    hists = pk.load(f)


def plot_acc_vs_loss(acc_ax, loss_ax, name, history):
    epochs = range(len(history.epoch)+1)
    acc_ax.plot(epochs, [0] + history.history['acc'])
    acc_ax.plot(epochs, [0] + history.history['val_acc'])
    acc_ax.set_title(f'{name} accuracy')

    loss_ax.plot(epochs, [1] + history.history['loss'])
    loss_ax.plot(epochs, [1] + history.history['val_loss'])
    loss_ax.set_title(f'{name} loss')


fig, axes = plt.subplots(len(hists), 2)
for (hist_name, hist_data), (ax1, ax2) in zip(hists.items(), axes):
    plot_acc_vs_loss(ax1, ax2, hist_name, hist_data)

plt.show()