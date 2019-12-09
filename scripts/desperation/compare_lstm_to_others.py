#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_prediction(ref_biases, predicted_biases, igu_pred_biases):
    plt.plot(ref_biases, 'b', label='referencyjne opoznienia')
    plt.plot(predicted_biases, 'r-.', label='LSTM')
    plt.plot(igu_pred_biases, 'k--', label='IGU-P')
    # plt.xlim([0, 96])
    plt.ylabel('Opoznienie [ns]')
    plt.xlabel('Epoka')
    # plt.yticks(np.arange(-4300, -4200, 10))
    # plt.xticks(np.arange(0, 97, 8))
    # plt.ylim([-4270, -4210])
    # plt.xlim([0, 96])
    plt.legend()
    # rcParams['figure.figsize'] = (5.5, 4.5)
    plt.show()


# Porownujemy wyniki z wartosciami referencyjnymi (IGU Observed) i oficjalnymi predykcjami (IGU Predicted)
def main(argv):
    
    # Set GPS number and directories with data
    lstm_file_name = argv[1]
    ref_file_name = argv[2]
    igu_pred_file_name = argv[3]
    
    # Read data from CSV
    lstm_pred_biases = pd.read_csv(lstm_file_name, sep=',',
                                   header=0, parse_dates=[0],
                                   index_col=0, squeeze=True).values
    ref_biases = pd.read_csv(ref_file_name, sep=';', header=0,
                             parse_dates=[0], index_col=0,
                             squeeze=True).values
    igu_pred_biases = pd.read_csv(igu_pred_file_name, sep=';',
                                  header=0, parse_dates=[0],
                                  index_col=0, squeeze=True).values
	
    # Calculate basic statistics
    lstm_mae = mean_absolute_error(ref_biases, lstm_pred_biases)
    lstm_mse = mean_squared_error(ref_biases, lstm_pred_biases)
    igu_p_mae = mean_absolute_error(ref_biases, igu_pred_biases)
    igu_p_mse = mean_squared_error(ref_biases, igu_pred_biases)
	
    print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('LSTM', lstm_mae, lstm_mse, sqrt(lstm_mse)))
    print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('IGU-P', igu_p_mae, igu_p_mse, sqrt(igu_p_mse)))
    
    # Plot results compared to reference value
    plot_prediction(ref_biases, lstm_pred_biases, igu_pred_biases)

if __name__ == '__main__':
    main(sys.argv)
