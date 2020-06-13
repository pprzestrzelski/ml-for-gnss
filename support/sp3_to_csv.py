import argparse
import os
import pandas as pd



def get_input_files(input_dir):
    files = {}
    for r, d, f in os.walk(input_dir):
        for file in f:
            if '.sp3' in file:
                files[file.split('.')[0]] = os.path.join(r, file)
    return files

def is_header_line(sp3_line):
    return sp3_line[0] in ['#','+','/','%']


def is_epoch_line(sp3_line):
    return sp3_line[0] == '*'


def is_satellite_line(sp3_line):
    return sp3_line[0] == 'P'

def is_prediction(sat_data):
    return sat_data['orbit_prediction'] or sat_data['clock_prediction']

def get_epoch_data(sp3_line):
    epoch_txt = sp3_line.split()
    year = epoch_txt[1]
    month = epoch_txt[2]
    day = epoch_txt[3]
    hour = epoch_txt[4]
    minutes = epoch_txt[5]

def get_satellite_data(sp3_line, epoch):
    pass


def sp3_to_pandas(file_name, convert_dates, include_predictions):
    data = []
    epoch = None
    with open(file_name, 'r') as sp3_file:
        for sp3_line in sp3_file:
            if is_header_line(sp3_line):
                continue # ignoring header lines
            elif is_epoch_line(sp3_line):
                epoch = get_epoch_data(sp3_line)
            elif is_satellite_line(sp3_line):
                sat_data = get_sattelite_data(sp3_line, epoch)
                if include_predictions or not is_prediction(sat_data):
                    data.append(sat_data)
            else:
                print(f'ERROR> UNKNOWN LINE TYPE IN FILE {file_name}')
                print(f'ERROR> {sp3_line}')
    dataframe = pd.DataFrame(data)


def check_range(files):
    pass
    
def main(input_dir, output_dir, convert_dates, include_predictions):
    files = get_input_files(input_dir)
    last = 0
    first = 2020
    for _, file_name in files.items():
        with open(file_name, 'r') as sp3_file:
            for sp3_line in sp3_file:
                if is_header_line(sp3_line):
                    continue # ignoring header lines
                elif is_epoch_line(sp3_line):
                    epoch_txt = sp3_line.split()
                    year = int(epoch_txt[1])
                    if year < first: first = year
                    if year > last: last = year
    print(f'First year : {first}')
    print(f'Last year : {last}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--convert_dates', action='store_true')
    parser.add_argument('-p', '--include_predictions', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.input_dir, args.output_dir, args.convert_dates, args.include_predictions)


