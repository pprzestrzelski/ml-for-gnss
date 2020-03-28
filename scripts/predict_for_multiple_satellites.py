import argparse
import os
from predict_clock_bias import predict_bias


def predict_for_multiple_sats(input_folder: str, model_dir: str, bias_column: str, epoch_column: str,
                              prediction_depth: int, first_epoch: float, epoch_step: float,
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
        predict_bias(files[i], args.models, args.bias_column_name, args.epoch_column_name,
                     args.prediction_depth, args.first_epoch, args.epoch_step, args.output_dir,
                     names[i])


def parse_arguments()-> argparse.ArgumentParser:
    desc = '''Script uses provided input data to teach a neural network'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_dir', help='directory with bias data')
    parser.add_argument('-m', '--models', help='directory that hold models for networks and preprocessor')
    parser.add_argument('-b', '--bias_column_name', help='name of clock bias column')
    parser.add_argument('-e', '--epoch_column_name', help='name of epoch column')
    parser.add_argument('-d', '--prediction_depth', help='how many entreis should be predicted', type=int)
    parser.add_argument('-f', '--first_epoch', help='first epoch in prediction', type=float)
    parser.add_argument('-s', '--epoch_step', help='name of epoch column', type=float)
    parser.add_argument('-o', '--output_dir', help='directory where output files will be saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    predict_for_multiple_sats(args.input_dir, args.models, args.bias_column_name, args.epoch_column_name,
                              args.prediction_depth, args.first_epoch, args.epoch_step, args.output_dir)
