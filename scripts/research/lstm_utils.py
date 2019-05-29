from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt


# create a differenced series
def diff(dataset, step=1):
    diffs = list()
    for i in range(step, len(dataset)):
        diffs.append(dataset[i] - dataset[i - step])
    return diffs


# invert differenced value
def inv_diff(last_obs, value):
    return value + last_obs


# scale train and test data to [-1, 1]
def scale(train):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = train.reshape(train.shape[0], 1)
    scaler = scaler.fit(train)
    # transform train
    train_scaled = scaler.transform(train)
    return scaler, train_scaled


# inverse scaling for a forecasted value
def inv_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


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
    plt.title('Strata trenowania i walidacji')
    plt.legend()
    plt.show()
