#!/usr/bin/env python

# Importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# noinspection DuplicatedCode
def main(argv):
    # Wczytujemy dane plik z danymi podany jako pierwszy argument w terminalu
    dataset = pd.read_csv(argv[1], sep=';')

    # Wyciągamy interesujące nas dane z zbioru danych, nazwa kolumny jest podana
    # jako drugi parametr
    time_series = dataset[argv[2]].to_numpy()

    # Skalujemy dane tak żeby nie wychodziły poza przedział <-1;1>
    scaler = MinMaxScaler(feature_range=(-1, 1))
    time_series_for_scaler = time_series.reshape(1, -1)  # Fit musi mieć tablicę 2D
    scaled_time_series = scaler.fit_transform(time_series_for_scaler)
    scaled_time_series = scaled_time_series.reshape(-1)  # I z powrotem do jednowymiarowej

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
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

    # Tworzymy sieć neuronową
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32,
                                   dropout=0.2,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False,
                                   input_shape=(None, inputs.shape[-1])
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
    model.fit(inputs, outputs, epochs=100, batch_size=32)

if __name__ == '__main__':
    main(sys.argv)
