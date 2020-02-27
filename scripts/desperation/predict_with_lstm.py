#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json


# Based on https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# noinspection DuplicatedCode
def predict_with_lstm(model, windowed_data, window_size, depth):
    predicted_data = []
    network_inputs = copy(windowed_data)
    
    while depth > 0:
        x = np.array(network_inputs.pop(0))
        y = model.predict(x.reshape(1, 1, window_size), verbose=0)

        if len(network_inputs) == 0:
            predictions.append(y)
            network_inputs.append(y)
            depth -= 1

    return predictions


# noinspection DuplicatedCode
def diff(dataset):
    diffs = list()
    for i in range(1, len(dataset)):
        diffs.append(dataset[i] - dataset[i - 1])
    return np.asarray(diffs)

def prepare_windowed_data(sat_name, column_name, input_dir,  window_size):
    input_file_name = os.path.join(input_dir, '{}.csv'.format(sat_name))
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

def load_networks(sat_name, networks_folder):
    models = {}
    weights = {}
    networks = {}
    for r, d, f in os.walk(networks_folder):
        for filename in f:
            file_info = filename.replace('.', '-').split('-')
            if len(file_info) == 4 and file_info[0] == sat_name:
                if file_info[3] == 'json':
                    models[file_info[1]] = os.path.join(r, filename)
                else:
                    weights[file_info[1]] = os.path.join(r, filename)

    for network_name in models.keys():
        model_json = None
        with open(models[network_name], 'r') as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights[network_name])
        model.compile(loss='mse', optimizer='rmsprop')
        networks[network_name].append(model)

    return networks

def build_dataframe_from_predictions(predictions, first_value, last_epoch, epoch_step, scale):
    predictions = np.asarray(lstm_predictions).flatten()
    predictions = predictions / scale
    bias = [first_value]
    for prediction in predictions:
        bias.append(bias[-1] + prediction)

    epochs = []
    for i in range(len(bias)):
        last_epoch += epoch_step
        prediction_epochs.append(last_epoch)

    dataframe = pd.DataFrame({'Epoch':prediction_epochs, 'Clock_bias':bias})
    dataframe = dataframe[['Epoch', 'Clock_bias']]
    return dataframe


def save_predictions(dataframe, sat_name, net_name, output_dir):
    file_path = os.path.join(output_dir, '{}_{}.csv'.format(sat_name, net_name))
    dataframe.to_csv(file_path, sep=';', index=False)

    
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
    sat_name = argv[1]
    column_name = argv[2]
    input_dir = argv[3]
    model_dir = argv[4]
    input_size = int(argv[5])
    prediction_depth = int(argv[6])
    scale = float(argv[7])
    output_dir = argv[8]
    last_epoch = float(argv[9])
    epoch_step = float(argv[10])
    
    windowed_data, first_value, start_epoch = prepare_windowed_data(sat_name, column_name, input_dir,  window_size)
    networks = load_networks(sat_name, networks_folder)
    
    for net_name, net_model in networks.items():
        predictions = predict_with_lstm(net_model, windowed_data, window_size, depth)
        dataframe = build_dataframe_from_predictions(predictions, first_value, last_epoch, epoch_step, scale)
        save_predictions(dataframe, sat_name, net_name, output_dir)


if __name__ == '__main__':
    main(sys.argv)
