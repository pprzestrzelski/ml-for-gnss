#!/usr/bin/env python
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

def main(argv):

    argc = len(argv)
    argc_desired = 9
    if argc != argc_desired:
        print("Wrong number of input arguments!")
        print("Usage: compare_lstm_to_others <PLIK_Z_DANYMI> <NAZWA_KOLUMNY> <TOPOLOGIA_SIECI_JSON> <PLIK_Z_WAGAMI> "
              "<ROZMIAR_WEJSCIA> <GLEBOKOSC_PREDYKCJI> <WSPOLCZYNNIK_SKALOWANIA> <PRN_SATELITY>")
        return

    # Dla trochę lepszej czytelności
    prediction_depth = int(argv[6])
    input_size = int(argv[5])
    scale = float(argv[7])

    # Wczytujemy dane plik z danymi podany jako pierwszy argument w terminalu
    dataset = pd.read_csv(argv[1], sep=';')

    # Wyciągamy interesujące nas dane z zbioru danych, nazwa kolumny jest podana
    # jako drugi parametr
    time_series = dataset[argv[2]].to_numpy()
    start_epoch = dataset['Epoch'][0]
    epoch_step = dataset['Epoch'][1] - start_epoch
    last_epoch = dataset['Epoch'].to_numpy()[-1]

    first_value = time_series[-1]
    
    # Zmieniamy szereg wartości w szereg różnic pomiędzy wartościami
    time_series = diff(time_series)

    # Wczytujemy topologię i parametry naszej sieci neuronowej
    model_json = None
    with open(argv[3], 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    # Doczytujemy do modelu wagi
    model.load_weights(argv[4])

    # Kompilujemy model, parametry ustawione na sztywno tak jak w skrypcie uczącym
    # paskudny antipattern
    model.compile(loss='mse', optimizer='rmsprop')

    # Zapisujemy predykcje z modelu LSTM !!!
    lstm_predictions = predict_with_lstm(model, time_series, scale,
                                         input_size, prediction_depth)

    bias = return_to_original_form(np.asarray(lstm_predictions).flatten(), first_value, scale)
    prediction_epochs = []
    for _ in bias:
        last_epoch += epoch_step
        prediction_epochs.append(last_epoch)
    
    print(time_series.shape)
    print(np.asarray(bias))
    print(np.asarray(prediction_epochs))

if __name__ == '__main__':
    main(sys.argv)
