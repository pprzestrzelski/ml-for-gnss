#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from keras.models import model_from_json
from matplotlib import pyplot as plt

#                             [1]             [2]                [3]                 [4]              [5]
# compare_lstm_to_others <PLIK_Z_DANYMI> <NAZWA_KOLUMNY> <TOPOLOGIA_SIECI_JSON> <PLIK_Z_WAGAMI> <ROZMIAR_WEJSCIA>
#                        <GLEBOKOSC_PREDYKCJI> <WSPOLCZYNNIK_SKALOWANIA> <PRN_SATELITY>
#                                 [6]                     [7]                 [8]


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


def plot_prediction(ref_biases, predicted_biases, igu_pred_biases, print_plot=False):
    plt.plot(predicted_biases, 'r-.', label='LSTM')
    plt.plot(igu_pred_biases, 'k--', label='IGU-P')
    plt.plot(ref_biases, 'b', label='referencyjne opóźnienia')
    # plt.xlim([0, 96])
    plt.ylabel('Opóźnienie [ns]')
    plt.xlabel('Epoka')
    # plt.yticks(np.arange(-4300, -4200, 10))
    # plt.xticks(np.arange(0, 97, 8))
    # plt.ylim([-4270, -4210])
    # plt.xlim([0, 96])
    plt.legend()
    # rcParams['figure.figsize'] = (5.5, 4.5)
    plt.show()


def main(argv):

    # Dla trochę lepszej czytelności
    prediction_depth = int(argv[6])
    input_size = int(argv[5])
    scale = float(argv[7])

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

    # Porownujemy wyniki z wartosciami referencyjnymi (IGU Observed) i oficjalnymi predykcjami (IGU Predicted)
    # UWAGA: ponizsze katalogi z odpowiednimi plikami trzeba dodac samodzielnie!
    gps_prn = argv[8]   # e.g. G05
    ref_file_name = "ref_data/igu_observed/{}.csv".format(gps_prn)  # ref data to test ANN
    igu_pred_file_name = "ref_data/igu_predicted/{}.csv".format(gps_prn)  # comparable predictions by IGU-P

    ref_biases = pd.read_csv(ref_file_name, sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True).values
    igu_pred_biases = pd.read_csv(igu_pred_file_name, sep=';', header=0,
                                  parse_dates=[0], index_col=0, squeeze=True).values

    plot_prediction(ref_biases, lstm_predictions, igu_pred_biases)


if __name__ == '__main__':
    main(sys.argv)
