import argparse
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


def compare_single_result(lstm_dir: str, igu: str, ref:str, out_file: str):
    IGU_COLUMN = 'igu'
    lstm_predictions = {}
    comparition = {}
    for r, d, f in os.walk(lstm_dir):
        for file in f:
            if '.csv' in file:
                filename = file.split('.')[0]
                satellite, network = filename.split('_')
                lstm_predictions[network]=(os.path.join(r, file))
    ref_biases = pd.read_csv(ref, sep=';', header=0, parse_dates=[0],
                             index_col=0, squeeze=True).values
    igu_pred_biases = pd.read_csv(igu, sep=';',header=0, parse_dates=[0],
                                  index_col=0, squeeze=True).values
    igu_p_mae = mean_absolute_error(ref_biases, igu_pred_biases)
    igu_p_mse = mean_squared_error(ref_biases, igu_pred_biases)
    print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('IGU-P', igu_p_mae, igu_p_mse, sqrt(igu_p_mse)))
    comparition[IGU_COLUMN] = igu_p_mae
    for network in lstm_predictions.keys():
        lstm_pred_biases = pd.read_csv(lstm_predictions[network], sep=';',
                                       header=0, parse_dates=[0],
                                       index_col=0, squeeze=True).values
        lstm_mae = mean_absolute_error(ref_biases, lstm_pred_biases)
        lstm_mse = mean_squared_error(ref_biases, lstm_pred_biases)
        print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('LSTM', lstm_mae, lstm_mse, sqrt(lstm_mse)))
        comparition[network] = lstm_mae


def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-l', '--lstm_dir', help='directory with lstm predictions')
    parser.add_argument('-i', '--igu', help='directory with igu predictions')
    parser.add_argument('-r', '--ref', help='directory with reference')
    parser.add_argument('-o', '--out_file', help='output file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    compare_single_result(args.lstm_dir, args.igu, args.ref, args.out_file)
