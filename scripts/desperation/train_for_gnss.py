#!/usr/bin/env python
# -*- coding: utf-8 -*-

# WYWOŁANIE
# train_for_gnss <PLIK_Z_DANYMI> <NAZWA_KOLUMNY> <ROZMIAR_WEJŚCIA_SIECI> <ILOŚĆ_EPOK> <STOSUNEK TRENING/TEST>
# <NAZWA_DLA_PLIKÓW_WYJŚCIOWYCH> <KATALOG_Z_WYJŚCIEM> <WSPÓŁCZYNNIK_SKALOWANIA>

# Importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import tensorflow as tf
from keras import regularizers
from math import floor
import os


# noinspection DuplicatedCode
def plot_lstm_loss(history, save_dir, file_name):
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
    file_name = '{}_loss.png'.format(file_name)
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    #plt.show()


def diff(dataset):
    diffs = list()
    for i in range(1, len(dataset)):
        diffs.append(dataset[i] - dataset[i - 1])
    return np.asarray(diffs)


# noinspection DuplicatedCode
def main(argv):
    # Wczytujemy dane plik z danymi podany jako pierwszy argument w terminalu
    dataset = pd.read_csv(argv[1], sep=';')

    # Wyciągamy interesujące nas dane z zbioru danych, nazwa kolumny jest podana
    # jako drugi parametr
    time_series = dataset[argv[2]].to_numpy()

    # Zmieniamy szereg wartości w szereg różnic pomiędzy wartościami
    time_series = diff(time_series)

    # Skalujemy dane tak żeby nie wychodziły poza przedział <-1;1>
    scaled_time_series = time_series * float(argv[8])


    # Tworzymy zestaw wejść i wyjść
    input_size = int(argv[3])
    inputs = []
    outputs = []
    for i in range(input_size, scaled_time_series.shape[0]):
        inputs.append(scaled_time_series[i - input_size:i])
        outputs.append(scaled_time_series[i])

    # Przekształcamy do tablic numpy
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    # Przekształcamy wejścia tak żeby pasowały do sieci LSTM
    #inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
    #outputs = np.reshape(outputs, (1, outputs.shape[0]))
    print('inputs = {}'.format(inputs.shape))
    print('outputs = {}'.format(outputs.shape))
    #sys.exit()


    # Rozdzielamy dane na treningowe i testowe
    train_coeff = float(argv[5])
    tr_count = int(floor(inputs.shape[0] * train_coeff))
    print('tr_count = {}'.format(tr_count))
    x_train = inputs[:tr_count, :]
    x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    y_train = outputs[:tr_count]
    x_test = inputs[tr_count:, :]
    x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
    y_test = outputs[tr_count:]

    # Tworzymy sieć neuronową
    print('x_train = {}'.format(x_train.shape))
    print('y_train = {}'.format(y_train.shape))
    #sys.exit()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32,
                                   dropout=0.2,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False,
                                   input_shape=(None, x_train.shape[-1])
                                   ))
    model.add(tf.keras.layers.LSTM(128,
                                   dropout=0.5,
                                   recurrent_dropout=0.5,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False
                                   ))
    model.add(tf.keras.layers.Dense(1,
                                    activation='linear'
                                    ))
    model.compile(loss='mse', optimizer='rmsprop')

    # Uczenie dla danych wejściowych i wyjściowych
    epochs = int(argv[4])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                        validation_data=(x_test, y_test), shuffle=False)

    # Zapisywanie wyników
    plot_lstm_loss(history, argv[7], argv[6])
    file_name = argv[6]
    file_name = '{}_model.json'.format(file_name)
    file_path = os.path.join(argv[7], file_name)
    model_json = model.to_json()
    with open(file_path, "w") as json_file:
        json_file.write(model_json)
    file_name = argv[6]
    file_name = '{}_weights.h5'.format(file_name)
    file_path = os.path.join(argv[7], file_name)
    model.save_weights(file_path)
    print('Model saved do not worry about exception.')


if __name__ == '__main__':
    try:
        main(sys.argv)
        print('Exception did not occured.')
    except Exception as e:
        print('Exception occured -> {}'.format(str(e)))
