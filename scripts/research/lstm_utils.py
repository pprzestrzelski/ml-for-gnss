from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 10
rcParams['figure.figsize'] = (5.5, 3.5)


class Scaler:
    def __init__(self):
        self.mean = 0.0
        self.scale = 1.0

    def do_scale(self, data):
        self.mean = np.mean(data)
        data -= self.mean
        self.scale = max(max(data), abs(min(data)))
        data /= self.scale
        return data

    def inverse_transform(self, data):
        data *= self.scale
        data += self.mean
        return data


# create a differenced series
def diff(dataset, step=1):
    diffs = list()
    for i in range(step, len(dataset)):
        diffs.append(dataset[i] - dataset[i - step])
    return diffs


# invert differenced value
def inv_diff(last_obs, value):
    return value + last_obs


# scale to [-1, 1]
def scale(train):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = train.reshape(train.shape[0], 1)
    scaler = scaler.fit(train)
    # transform train
    train_scaled = scaler.transform(train)
    return scaler, train_scaled


def create_lstm_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# Accepts simple, one-dimensional array as an input
def generate_lstm_datasets(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i + look_back)]
        data_x.append(a)
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)


def plot_lstm_loss(history, print_plot=False):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość funkcji straty')
    plt.legend()

    plt.xlim([1, len(loss)])
    plt.xticks(np.arange(0, len(loss)+10, 10))

    rcParams['figure.figsize'] = (5.5, 3)
    if print_plot:
        plt.savefig("__loss.png", bbox_inches='tight')
    plt.show()


def plot_raw_data(data, print_plot=False):
    plt.plot(data, 'k')
    epochs = len(data)
    logging.info("Plot GNSS clock data")
    plt.xlabel('Epoka')
    plt.ylabel('Opóźnienie [ns]')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-6000, -3000, 200))
    plt.xlim([0, epochs])
    plt.ylim([-5400, -4200])

    rcParams['figure.figsize'] = (5.5, 3.5)
    if print_plot:
        plt.savefig("__raw.png", bbox_inches='tight')
    plt.show()


def plot_differences(data, print_plot=False):
    plt.plot(data, 'k')
    epochs = len(data)
    logging.info("Plot differences")
    logging.info("Max diff: {}".format(max(data)))
    logging.info("Min diff: {}".format(min(data)))
    plt.xlabel('Epoka')
    plt.ylabel('Różnica opóźnień [ns]')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-3, 3.01, 0.5))
    plt.xlim([0, epochs])
    plt.ylim([-3, 3])

    rcParams['figure.figsize'] = (5.5, 3.5)
    if print_plot:
        plt.savefig("__diff.png", bbox_inches='tight')
    plt.show()


def plot_scaled_values(data, print_plot=False):
    logging.info("Plot scaled data")
    logging.info("Max scaled:", max(data))
    logging.info("Min scaled:", min(data))
    logging.info("Mean value:", np.mean(data))
    logging.info("Std dev:", np.std(data))
    plt.plot(data, 'k')
    epochs = len(data)
    plt.xlabel('Epoka')
    plt.ylabel('Opóźnienie znormalizowane')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-3, 3.01, 0.5))
    plt.xlim([0, epochs])
    plt.ylim([-3, 3])

    rcParams['figure.figsize'] = (5.5, 3.5)
    if print_plot:
        plt.savefig("__norm_diff.png", bbox_inches='tight')
    plt.show()


def plot_prediction(ref_biases, predicted_biases, igu_pred_biases, print_plot=False):
    plt.plot(predicted_biases, 'r-.', label='LSTM')
    plt.plot(igu_pred_biases, 'k--', label='IGU-P')
    plt.plot(ref_biases, 'b', label='referencyjne opóźnienia')
    plt.xlim([0, 96])
    plt.ylabel('Opóźnienie [ns]')
    plt.xlabel('Epoka')
    plt.yticks(np.arange(-4300, -4200, 10))
    plt.xticks(np.arange(0, 97, 8))
    plt.ylim([-4270, -4210])
    plt.xlim([0, 96])
    plt.legend()

    rcParams['figure.figsize'] = (5.5, 4.5)
    if print_plot:
        plt.savefig("__pred.png", bbox_inches='tight')
    plt.show()


def plot_prediction_error(lstm, igu_p, linear, poly_2, poly_4, poly_8, print_plot=False):
    plt.plot(lstm, 'r', label='LSTM')
    plt.plot(igu_p, 'k', label='IGU-P')
    plt.plot(linear, 'b', label='Liniowa')
    plt.plot(poly_2, 'y', label='Wielomianowa 2-ego stopnia')
    plt.plot(poly_4, 'm', label='Wielomianowa 4-ego stopnia')
    plt.plot(poly_8, 'c', label='Wielomianowa 8-ego stopnia')
    plt.ylabel('Błąd predykcji [ns]')
    plt.xlabel('Epoka')
    plt.yticks(np.arange(0, 3.01, 0.5))
    plt.xticks(np.arange(0, 97, 8))
    plt.ylim([0, 3])
    plt.xlim([0, 96])
    plt.legend()

    rcParams['figure.figsize'] = (5.5, 4.5)
    if print_plot:
        plt.savefig("__pred_error.png", bbox_inches='tight')
    plt.show()
