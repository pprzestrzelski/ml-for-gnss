#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from keras.models import model_from_json

# train_for_gnss <PLIK_Z_DANYMI> <NAZWA_KOLUMNY> <PLIK_Z_MODELEM> <PLIK_Z_WAGAMI> <KATALOG_DLA_WYJŚĆ>
# <SKALOWANIE (NORMALIZACJA)> <ROZMIAR_WEJŚCIA_SIECI> <GŁĘBOKOŚĆ_PREDYKCJI>

# Based on https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# noinspection DuplicatedCode
def predict_with_lstm(model, time_series, scale, window_size, depth):
    predictions = []
    time_series = time_series / scale
    windowed_data = list(time_series[-window_size:])

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


def main(argv):

    # Dla trochę lepszej czytelności
    prediction_depth = int(argv[6])
    input_size = int(argv[5])
    scale = float(argv[4])

    # Wczytujemy dane plik z danymi podany jako pierwszy argument w terminalu
    dataset = pd.read_csv(argv[1], sep=';')

    # Wyciągamy interesujące nas dane z zbioru danych, nazwa kolumny jest podana
    # jako drugi parametr
    time_series = dataset[argv[2]].to_numpy()

    # Zmieniamy szereg wartości w szereg różnic pomiędzy wartościami
    time_series = diff(time_series)

    # Wczytujemy topologię i parametry naszej sieci neuronowej
    model_json = None
    with open(argv[3], 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Doczytujemy do modelu wagi
    model.load_weights(argv[4])

    # Kompilujemy model, parametry ustawione na sztywno tak jak w skrypcie uczącym
    # paskudny antipattern
    model.compile(loss='mse', optimizer='rmsprop')

    # Zapisujemy predykcje z modelu LSTM !!!
    lstm_predictions = predict_with_lstm(model, time_series, scale,
                                         input_size, prediction_depth)




if __name__ == '__main__':
    main(sys.argv)