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
from network_models import build_models

def diff(dataset):
    diffs = list()
    for i in range(1, len(dataset)):
        diffs.append(dataset[i] - dataset[i - 1])
    return np.asarray(diffs)

def build_output_file_path(output_dir, sat_name, model_name, suffix, ext):
    file_name = '{}_{}_{}.{}'.format(sat_name, model_name, suffix, ext)
    return os.path.join(output_dir, file_name)

def save_network_history(history, model_name, sat_name, output_dir):
    hist_df = pd.DataFrame.from_dict(history.history, orient="index")
    loss = hist_df.to_csv(build_output_file_path(output_dir, sat_name,
                                                 model_name, 'history',
                                                 'csv'))
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Wartość funkcji straty')
    plt.legend()

    plt.xlim([1, len(loss)])
    plt.xticks(np.arange(0, len(loss)+10, 10))

    rcParams['figure.figsize'] = (5.5, 3)
    file_path = build_output_file_path(output_dir, sat_name, model_name,
                                       'history', 'png')
    plt.savefig(file_path, bbox_inches='tight')


def train_networks(input_size, epochs, x_train, y_train, x_test, y_test):
    models = build_models(input_size, (None, x_train.shape[-1]))
    histories = {}
    
    for name, model in models.items():
        try:
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                            validation_data=(x_test, y_test), shuffle=False)
            histories[name] = history
        except Exception as e:
            print('Exception during training.')
    return models, histories

def prepare_data(csv_file_name, column_name, scale, input_size, train_coefficent):
    dataset = pd.read_csv(csv_file_name, sep=';')
    time_series = dataset[column_name].to_numpy()
    time_series = diff(time_series)
    time_series = time_series * scale

    inputs = []
    outputs = []
    for i in range(input_size, time_series.shape[0]):
        inputs.append(time_series[i - input_size:i])
        outputs.append(time_series[i])
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    tr_count = int(floor(inputs.shape[0] * train_coefficent))
    x_train = inputs[:tr_count, :]
    x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    y_train = outputs[:tr_count]
    x_test = inputs[tr_count:, :]
    x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
    y_test = outputs[tr_count:]

    return x_train, y_train, x_test, y_test

def save_outputs(sat_name, output_dir, models, histories):
    for model_name in models.keys():
        try:
            # save_network_history(histories[model_name], model_name, sat_name, output_dir)
            model_json = models[model_name].to_json()
            file_path = build_output_file_path(output_dir, sat_name, model_name,
                                               'model', 'json')
            with open(file_path, "w") as json_file:
                json_file.write(model_json)
                file_path = build_output_file_path(output_dir, sat_name, model_name,
                                                   'weights', 'h5')
            models[model_name].save_weights(file_path)
        except Exception as e:
            raise e
            print('Exception during saving -> {}'.format(str(e)))
    
def main(argv):
    csv_file_name = argv[1]
    column_name = argv[2]
    input_size = int(argv[3])
    epochs = int(argv[4])
    train_coefficent = float(argv[5])
    sat_name = argv[6]
    output_dir = argv[7]
    scale = float(argv[8])

    
    x_train, y_train, x_test, y_test = prepare_data(csv_file_name, column_name,
                                                    scale, input_size,
                                                    train_coefficent)
    models, histories = train_networks(input_size, epochs, x_train, y_train,
                                       x_test, y_test)
    save_outputs(sat_name, output_dir, models, histories)
    


if __name__ == '__main__':
    try:
        main(sys.argv)
        print('Exception did not occured.')
    except Exception as e:
        raise e
        print('Exception occured -> {}'.format(str(e)))
