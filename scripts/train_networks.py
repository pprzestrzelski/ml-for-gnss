import pandas as pd
import numpy as np
import argparse
from support import DataPrerocessing
from network_builder import build_models
import sys
import os

def train_networks_core(input_size, epochs, x_train, y_train, x_test, y_test):
    models = build_models(input_size, (None, x_train.shape[-1]))
    histories = {}
    
    for name, model in models.items():
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                            validation_data=(x_test, y_test), shuffle=False)
        histories[name] = history
        
    return models, histories


def build_output_file_path(output_dir, sat_name, model_name, suffix, ext):
    file_name = '{}_{}_{}.{}'.format(sat_name, model_name, suffix, ext)
    return os.path.join(output_dir, file_name)


def save_outputs(sat_name, output_dir, models, histories, preprocessor):
    preprocessor_file = '{}_preprocessor.json'.format(sat_name)
    preprocessor_file = os.path.join(output_dir, preprocessor_file)
    preprocessor.to_json(preprocessor_file)
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


def train_networks(csv_file_name: str, bias_column_name: str, epoch_column_name: str,
                   input_size: int, epochs: int, train_coefficent: float,
                   sat_name: str, output_dir: str):
    dataframe = pd.read_csv(csv_file_name, sep=';')
    bias = dataframe[bias_column_name].to_numpy()
    clock_epochs = dataframe[epoch_column_name].to_numpy()
    
    preprocessor = DataPrerocessing(training_coefficent=train_coefficent)
    processed = preprocessor.fit_transform(bias, clock_epochs, False)
    x, y = preprocessor.prepare_windowed_data(processed)
    x_train, y_train, x_test, y_test = preprocessor.split_training_and_validation(x, y)
    x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
    models, histories = train_networks_core(input_size, epochs, x_train, y_train, x_test, y_test)
    save_outputs(sat_name, output_dir, models, histories, preprocessor)


def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input',
                        help='csv with clock bias',
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
    parser.add_argument('-l', '--input_size',
                        help='length of neural network input vector',
                        type=int,
                        default=32)
    parser.add_argument('-e', '--epochs',
                        help='how many epochs should be used for training',
                        type=int,
                        default=10)
    parser.add_argument('-t', '--train_coefficent',
                        help='part of data that will be used for training (from 0 to 1)',
                        type=float,
                        default=0.8)
    parser.add_argument('-n', '--sat_name',
                        help='name of satellite that will be used in output files',
                        type=str,
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='directory where output files will be saved',
                        type=str,
                        required=True)
    parser.add_argument('-s', '--scale',
                        help='when set forces a given scaling factor',
                        type=float,
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    train_networks(args.input, args.bias_column_name, args.epoch_column_name,
                   args.input_size, args.epochs, args.train_coefficent,
                   args.sat_name, args.output_dir)


