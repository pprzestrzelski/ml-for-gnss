from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 10


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


def plot_lstm_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość straty')
    plt.legend()

    plt.xlim([0, len(loss)])
    plt.xticks(np.arange(0, len(loss), 10))

    plt.show()


def plot_raw_data(data):
    plt.plot(data, 'k')
    epochs = len(data)
    print("Plot GNSS clock data")
    plt.xlabel('Epoka')
    plt.ylabel('Opóźnienie [ns]')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-6000, -3000, 200))
    plt.xlim([0, epochs])
    plt.ylim([-5400, -4200])

    plt.show()


def plot_differences(data):
    plt.plot(data, 'k')
    epochs = len(data)
    print("Plot differences")
    print("Max diff:", max(data))
    print("Min diff:", min(data))
    plt.xlabel('Epoka')
    plt.ylabel('Różnica opóźnień [ns]')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-3, 3.01, 0.5))
    plt.xlim([0, epochs])
    plt.ylim([-3, 3])

    plt.show()


def plot_scaled_values(data):
    print("Plot scaled data")
    print("Max scaled:", max(data))
    print("Min scaled:", min(data))
    print("Mean value:", np.mean(data))
    print("Std dev:", np.std(data))
    plt.plot(data, 'k')
    epochs = len(data)
    plt.xlabel('Epoka')
    plt.ylabel('Wartość znormalizowana')
    plt.xticks(np.arange(0, epochs, 192))
    plt.yticks(np.arange(-3, 3.01, 0.5))
    plt.xlim([0, epochs])
    plt.ylim([-3, 3])

    plt.show()


def plot_prediction(ref_biases, predicted_biases, igu_pred_biases):
    plt.plot(predicted_biases, 'r-.', label='LSTM')
    plt.plot(igu_pred_biases, 'k--', label='IGU-P')
    plt.plot(ref_biases, 'b', label='referencyjne opóźnienia')
    plt.xlim([0, 96])
    plt.ylabel('[ns]')
    plt.xlabel('Epoka')
    plt.yticks(np.arange(-4300, -4200, 10))
    plt.xticks(np.arange(0, 96, 8))
    plt.ylim([-4270, -4210])
    plt.xlim([0, 96])
    plt.legend()
    plt.show()


def plot_prediction_error(lstm, igu_p):
    plt.plot(lstm, 'r', label='LSTM')
    plt.plot(igu_p, 'b', label='IGU-P')
    plt.ylabel('[ns]')
    plt.xlabel('Epoka')
    plt.yticks(np.arange(0, 3, 0.5))
    plt.xticks(np.arange(0, 96, 8))
    plt.ylim([0, 3])
    plt.xlim([0, 96])
    plt.legend()
    plt.show()
