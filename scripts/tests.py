import pandas as pd
import numpy as np
import argparse

from support import DataPrerocessing

def parse_arguments()-> argparse.ArgumentParser:
    desc = 'Test functionalities in project'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--csv', help='csv file used for tests')
    return parser.parse_args()



def main(csv_file: str):
    df = pd.read_csv(csv_file, sep=';')
    proc = DataPrerocessing()
    bias = df['Clock_bias'].to_numpy()
    epochs = df['Epoch'].to_numpy()
    trans = proc.fit_transform(bias, epochs, False)
    print('='*90)
    print(trans)

if __name__ == '__main__':
    args = parse_arguments()
    main(args.csv)
