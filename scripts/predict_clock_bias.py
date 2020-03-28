import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from support import DataPrerocessing
import sys
import os


def load_models(sat_name:str, model_dir: str):
    preprocessor_file = os.path.join(model_dir,'{}_preprocessor.json'.format(sat_name))
    print('Loading preprocessor from {}'.format(preprocessor_file))
    preprocessor = DataPrerocessing.load_json(preprocessor_file)

    models = {}
    weights = {}
    for r, d, f in os.walk(model_dir):
        for filename in f:
            file_info = filename.replace('.', '_').split('_')
            if len(file_info) == 4 and file_info[0] == sat_name:
                if file_info[3] == 'json':
                    models[file_info[1]] = os.path.join(r, filename)
                else:
                    weights[file_info[1]] = os.path.join(r, filename)

    networks = {}
    for network_name in models.keys():
        model_json = None
        print('Compiling network : {}'.format(network_name))
        print('Loading model from {}'.format(models[network_name]))
        with open(models[network_name], 'r') as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        print('Loading weights from {}'.format(weights[network_name]))
        model.load_weights(weights[network_name])
        networks[network_name] = model

    return preprocessor, networks


def predict_bias(in_file: str, model_dir: str, bias_column: str, epoch_column: str,
                 prediction_depth: int, first_epoch: float, epoch_step: float,
                 output_dir: str, sat_name:str):
    dataframe = pd.read_csv(in_file, sep=';')
    bias = dataframe[bias_column].to_numpy()
    preprocessor, networks = load_models(sat_name, model_dir)
    bias = preprocessor.fit_transform_bias(bias, True)
    x, y = preprocessor.prepare_windowed_data(bias)
    x_train, y_train, x_test, y_test = preprocessor.split_training_and_validation(x, y)


def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help='csv with clock bias')
    parser.add_argument('-m', '--models', help='directory that hold models for networks and preprocessor')
    parser.add_argument('-b', '--bias_column_name', help='name of clock bias column')
    parser.add_argument('-e', '--epoch_column_name', help='name of epoch column')
    parser.add_argument('-d', '--prediction_depth', help='how many entreis should be predicted', type=int)
    parser.add_argument('-f', '--first_epoch', help='first epoch in prediction', type=float)
    parser.add_argument('-s', '--epoch_step', help='name of epoch column', type=float)
    parser.add_argument('-o', '--output_dir', help='directory where output files will be saved')
    parser.add_argument('-n', '--sat_name', help='name of satellite used for generating output files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    predict_bias(args.input, args.models, args.bias_column_name, args.epoch_column_name,
                 args.prediction_depth, args.first_epoch, args.epoch_step, args.output_dir,
                 args.sat_name)
