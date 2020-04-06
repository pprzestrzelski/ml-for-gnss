import argparse
import os
from train_networks import train_networks


def train_multiple_networks(input_folder: str, bias_column_name: str, epoch_column_name: str,
                            input_size: int, epochs: int, train_coefficent: float,
                            output_dir: str):
    files = []
    names = []
    for r, d, f in os.walk(input_folder):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
                names.append(file.split('.')[0])

    for i in range(len(files)):
        print('{} -> {}'.format(names[i], files[i]))
        train_networks(files[i], bias_column_name, epoch_column_name,
                       input_size, epochs, train_coefficent,
                       names[i], output_dir)


def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_directory',
                        help='csv with clock bias',
                        type=str,
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='directory where output files will be saved',
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
    parser.add_argument('-s', '--scale',
                        help='when set forces a given scaling factor',
                        type=float,
                        default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train_multiple_networks(args.input_directory, args.bias_column_name, args.epoch_column_name,
                            args.input_size, args.epochs, args.train_coefficent, args.output_dir)
