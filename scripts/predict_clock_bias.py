import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from support import DataPrerocessing
import sys
import os
import copy

def load_models(sat_name:str, model_dir: str):
    preprocessor_file = os.path.join(model_dir,'{}_preprocessor.json'.format(sat_name))
    print('Loading preprocessor from {}'.format(preprocessor_file))
    preprocessor = DataPrerocessing.load_json(preprocessor_file)
    print('Preprocessor loaded with mean={} and scale={}'.format(preprocessor.mean, preprocessor.scale))
    models = {}
    weights = {}
    for r, d, f in os.walk(model_dir):
        for filename in f:
            file_info = filename.replace('.', '_').split('_')
            if 'preprocessor' in file_info: continue
            if file_info[0] == sat_name:
                if file_info[-1] == 'json':
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


def prediction_core(model, windowed_data, window_size, depth):
    predicted_data = []
    network_inputs = windowed_data.tolist()

    while depth > 0:
        x = np.array(network_inputs.pop(0))
        y = model.predict(x.reshape(1, 1, window_size), verbose=0)

        if len(network_inputs) == 0:
            predicted_data.append(y)
            window = np.delete(x, 0, 0)
            window= np.append(window, y)
            network_inputs.append(window)
            depth -= 1

    return predicted_data


def build_dataframe_from_predictions(predictions, first_value, last_epoch, epoch_step, scale, mean):
    predictions = np.asarray(predictions).flatten()
    predictions = predictions / scale
    predictions += mean
    bias = [first_value]
    for prediction in predictions:
        bias.append(bias[-1] + prediction)

    epochs = []
    for i in range(len(bias)):
        last_epoch += epoch_step
        epochs.append(last_epoch)

    dataframe = pd.DataFrame({'Epoch':epochs, 'Clock_bias':bias})
    dataframe = dataframe[['Epoch', 'Clock_bias']]
    return dataframe


def predict_bias(in_file: str, model_dir: str, bias_column: str, epoch_column: str,
                 prediction_depth: int, first_epoch: float, epoch_step: float,
                 output_dir: str, sat_name:str):
    dataframe = pd.read_csv(in_file, sep=';')
    bias = dataframe[bias_column].to_numpy()
    preprocessor, networks = load_models(sat_name, model_dir)
    bias = preprocessor.fit_transform_bias(bias, True)
    x, y = preprocessor.prepare_windowed_data(bias)
    for net_name, network in networks.items():
        predictions = prediction_core(network, x, preprocessor.window_size, prediction_depth)
        df = build_dataframe_from_predictions(predictions, preprocessor.initial_bias, first_epoch, epoch_step,
                                              preprocessor.scale, preprocessor.mean)
        out_file = os.path.join(output_dir,'{}_{}.csv'.format(sat_name, net_name))
        df.to_csv(out_file, sep=';', index=False)
    

def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input',
                        help='csv with clock bias',
                        type=str,
                        required=True)
    parser.add_argument('-m', '--models',
                        help='directory that hold models for networks and preprocessor',
                        type=str,
                        required=True)
    parser.add_argument('-b', '--bias_column_name',
                        help='name of clock bias column',
                        type=str,
                        default='Clock_bias')
    parser.add_argument('-c', '--epoch_column_name',
                        help='name of epoch column',
                        type=str,
                        default='Epoch')
    parser.add_argument('-d', '--prediction_depth',
                        help='how many entreis should be predicted',
                        type=int,
                        default=95)
    parser.add_argument('-f', '--first_epoch',
                        help='first epoch in prediction',
                        type=float,
                        default=2010.0)
    parser.add_argument('-e', '--epoch_step',
                        help='step between epochs',
                        type=float,
                        default=0.001488095238)
    parser.add_argument('-o', '--output_dir',
                        help='directory where output files will be saved',
                        type=str,
                        required=True)
    parser.add_argument('-n', '--sat_name',
                        help='name of satellite used for generating output files',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    predict_bias(args.input, args.models, args.bias_column_name, args.epoch_column_name,
                 args.prediction_depth, args.first_epoch, args.epoch_step, args.output_dir,
                 args.sat_name)
