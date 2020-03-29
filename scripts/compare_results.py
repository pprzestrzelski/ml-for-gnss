import argparse
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def prepare_files(lstm_dir: str, igu_dir: str, ref_dir: str):
    references = {}
    igu_predictions = {}
    lstm_predictions = {}
    
    for r, d, f in os.walk(ref_dir):
        for file in f:
            if '.csv' in file:
                satellite = file.split('.')[0]
                references[satellite]=(os.path.join(r, file))
                
    for r, d, f in os.walk(igu_dir):
        for file in f:
            if '.csv' in file:
                satellite = file.split('.')[0]
                igu_predictions[satellite]=(os.path.join(r, file))

    for r, d, f in os.walk(lstm_dir):
        for file in f:
            if '.csv' in file:
                filename = file.split('.')[0]
                satellite, network = filename.split('_')
                try:
                    lstm_predictions[satellite][network]=(os.path.join(r, file))
                except KeyError:
                    lstm_predictions[satellite] = {}
                    lstm_predictions[satellite][network]=(os.path.join(r, file))
                    
    return references, igu_predictions, lstm_predictions


def compare_results(lstm_dir: str, igu_dir: str, ref_dir: str):
    references, igu_predictions, lstm_predictions = prepare_files(lstm_dir, igu_dir, ref_dir)
    for satellite in references.keys():
        ref_biases = pd.read_csv(references[satellite], sep=';', header=0,
                                 parse_dates=[0], index_col=0,
                                 squeeze=True).values
        igu_pred_biases = pd.read_csv(igu_predictions[satellite], sep=';',
                                      header=0, parse_dates=[0],
                                      index_col=0, squeeze=True).values
        igu_p_mae = mean_absolute_error(ref_biases, igu_pred_biases)
        igu_p_mse = mean_squared_error(ref_biases, igu_pred_biases)
        print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('IGU-P', igu_p_mae, igu_p_mse, sqrt(igu_p_mse)))

        for network in lstm_predictions[satellite].keys():
            lstm_pred_biases = pd.read_csv(lstm_predictions[satellite][network], sep=';',
                                           header=0, parse_dates=[0],
                                           index_col=0, squeeze=True).values
            lstm_mae = mean_absolute_error(ref_biases, lstm_pred_biases)
            lstm_mse = mean_squared_error(ref_biases, lstm_pred_biases)
            print("%10s => MAE = %.4f MSE = %.4f RMS = %.4f" % ('LSTM', lstm_mae, lstm_mse, sqrt(lstm_mse)))

def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-l', '--lstm_dir', help='directory with lstm predictions')
    parser.add_argument('-i', '--igu_dir', help='directory with igu predictions')
    parser.add_argument('-r', '--ref_dir', help='directory with reference')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    compare_results(args.lstm_dir, args.igu_dir, args.ref_dir)
