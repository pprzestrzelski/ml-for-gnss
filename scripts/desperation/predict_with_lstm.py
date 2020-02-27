#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json


# Based on https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# noinspection DuplicatedCode
def predict_with_lstm(model, time_series, scale, window_size, depth):
    predictions = []
    time_series = time_series / scale
    windowed_data = list(time_series[-window_size:])

    print(window_size)
    #sys.exit()
    # predict!
    for _ in range(depth):
        predictioner = np.array(windowed_data)
        yhat = model.predict(predictioner.reshape(1, 1, window_size), verbose=0)

        # add to the memory
        predictions.append(yhat)

        # prepare window for next prediction with one new prediction
        windowed_data.append(yhat)
        windowed_data.pop(0)

    return predictions


# noinspection DuplicatedCode
def diff(dataset):
    diffs = list()
    for i in range(1, len(dataset)):
        diffs.append(dataset[i] - dataset[i - 1])
    return np.asarray(diffs)

def return_to_original_form(diffs, first_value, scale):
    diffs = diffs / scale
    bias = [first_value]
    for d in diffs:
        bias.append(bias[-1]+d)
    return bias


def prepare_windowed_data(input_file_name, column_name, window_size):
    dataset = pd.read_csv(input_file_name, sep=';')
    time_series = dataset[column_name].to_numpy()
    start_epoch = dataset['Epoch'][0]
    first_value = time_series[0]    
    time_series = diff(time_series)
    time_series = time_series / scale

    endpoint = len(time_series)
    startpoint = endpoint - window_size
    windowed_data = []
    while startpoint >= 0 :
        windowed_data.append(time_series[startpoint:endpoint])
        endpoint = startpoint
        startpoint = endpoint - window_size

    return windowed_data, first_value, start_epoch



def main(argv):

    argc = len(argv)
    argc_desired = 11
    if argc != argc_desired:
        print("Wrong number of input arguments! Expected {} got {}.".format(argc_desired, argc))
        print("Usage: compare_lstm_to_others <PLIK_Z_DANYMI> <NAZWA_KOLUMNY> <TOPOLOGIA_SIECI_JSON> <PLIK_Z_WAGAMI> "
              "<ROZMIAR_WEJSCIA> <GLEBOKOSC_PREDYKCJI> <WSPOLCZYNNIK_SKALOWANIA> <PLIK_WYJŚCIOWY>"
              "<EPOKA_ZEROWA_PREDYKCJI> <KROK_CZASOWY_PREDYKCJI>")
        return

    # Dla trochę lepszej czytelności
    input_file_name = argv[1]
    column_name = argv[2]
    topology_file_name = argv[3]
    weights_file_name = argv[4]
    input_size = int(argv[5])
    prediction_depth = int(argv[6])
    scale = float(argv[7])
    output_file_name = argv[8]
    last_epoch = float(argv[9])
    epoch_step = float(argv[10])

    windowed_data, first_value, start_epoch = prepare_windowed_data(input_file_name, column_name, window_size)

    # Wczytujemy topologię i parametry naszej sieci neuronowej
    model_json = None
    with open(topology_file_name, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    # Doczytujemy do modelu wagi
    model.load_weights(weights_file_name)

    # Kompilujemy model, parametry ustawione na sztywno tak jak w skrypcie uczącym
    # paskudny antipattern
    model.compile(loss='mse', optimizer='rmsprop')

    # Zapisujemy predykcje z modelu LSTM !!!
    lstm_predictions = predict_with_lstm(model, time_series, scale,
                                         input_size, prediction_depth)

    bias = return_to_original_form(np.asarray(lstm_predictions).flatten(), first_value, scale)
    prediction_epochs = []
    for i in range(len(bias)):
        print('DEPTH = {} EPOCH = {} STEP = {}'.format(i, last_epoch, epoch_step))
        last_epoch += epoch_step
        prediction_epochs.append(last_epoch)
    
    print(time_series.shape)
    data = {'Epoch':prediction_epochs, 'Clock_bias':bias}
    dataframe = pd.DataFrame(data)
    dataframe = dataframe[['Epoch', 'Clock_bias']]
    print(dataframe.head())
    dataframe.to_csv(output_file_name, sep=';', index=False)

if __name__ == '__main__':
    main(sys.argv)
