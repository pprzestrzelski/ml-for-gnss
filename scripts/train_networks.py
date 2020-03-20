import pandas as pd
import numpy as np
import argparse
from support import DataPrerocessing



def train_networks(csv_file_name: str, bias_column_name: str, epoch_column_name: str,
                   input_size: int, epochs: int, train_coefficent: float,
                   sat_name: str, output_dir: str, scale: float):
    dataframe = pd.read_csv(csv_file_name, sep=';')
    bias = dataframe[bias_column_name].as_numpy()
    epochs = dataframe[epoch_column_name].as_numpy()
    
    preprocessor = DataPrerocessing()
    processed = preprocessor.fit_transform(bias, epochs, False)
    
    

def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help='csv with clock bias')
    parser.add_argument('-b', '--bias_column_name', help='name of clock bias column')
    parser.add_argument('-c', '--epoch_column_name', help='name of epoch column')
    parser.add_argument('-l', '--input_size', help='length of neural network input vector')
    parser.add_argument('-e', '--epochs', help='how many epochs should be used for training')
    parser.add_argument('-t', '--train_coefficent', help='part of data that will be used for training (from 0 to 1)')
    parser.add_argument('-n', '--sat_name', help='name of satellite that will be used in output files')
    parser.add_argument('-o', '--output_dir', help='directory where output files will be saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    

    main()


