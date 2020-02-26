#!/usr/bin/python3

import sys
import os
import pandas as pd
import numpy as np
import datetime
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_DATA_LEN = 96   # TODO: set as an command line argument, not global var
DEFAULT_COMP_LABEL = 'IGU-P'    # TODO: the same as above!

SATELLITES = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', \
    'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', \
        'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24', \
            'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31',  'G32']
SATS_TO_OMMIT = []
ERROR_TYPES = ['DIFF', 'MAE', 'MSE', 'RMS']


def output_errors(errors, error_type, output_dir):
    time_format = '%d-%m-%Y_%H:%M:%S'
    curr_time = datetime.datetime.now().strftime(time_format)
    file_name = curr_time + '_' + error_type + '.csv'
    errors.to_csv(output_dir + '/' + file_name, sep=';')


def get_error(ref_data, data, function):
    return function(ref_data, data)


def rms_error(ref_data, data):
    mse = mean_squared_error(ref_data, data)
    return sqrt(mse)


def calculate_errors(error_type, ref_data, pred_data, comp_data):
    labels = []
    sats_done = []
    errors = []
    pred_labels = []
    
    # Find all prediction names
    for s in SATELLITES:
        if s in pred_data:
            pred_d = pred_data[s]
            for key in pred_d.keys():
                pred_labels.append(key)
            pred_labels.sort()
            break   # FIXME: gets labels for the first satellite only, 
                    #        may be error prone! 

    # Determine error function
    if error_type == 'DIFF':
        error_function = lambda x, y: np.mean((np.array(y) - np.array(x)))
    elif error_type == 'MAE':
        error_function = mean_absolute_error
    elif error_type == 'MSE':
        error_function = mean_squared_error
    elif error_type == 'RMS':
        error_function = rms_error

    print('No results for satellites: {}'.format(SATS_TO_OMMIT))
    for sat in SATELLITES:
        if sat not in SATS_TO_OMMIT:
            sat_err = []

            # Calculation of comparable product errors
            if DEFAULT_COMP_LABEL not in labels:
                labels.append(DEFAULT_COMP_LABEL)
            sat_err.append(get_error(ref_data[sat], comp_data[sat], \
                error_function))         

            # Get errors of our predictions
            sat_pred = pred_data[sat]
            for pred_type in pred_labels:
                if pred_type not in labels:
                    labels.append(pred_type)
                sp = sat_pred[pred_type]
                sat_err.append(get_error(ref_data[sat], sp, \
                    error_function))
            
            sats_done.append(sat)
            errors.append(sat_err)

    data_frame = pd.DataFrame(data=errors, columns=labels, \
        index=sats_done, dtype=float)
    # Gives a name to an index column
    data_frame.index.name = 'Sats'

    return data_frame


# reads: G01_LSTM1.csv, G01_LSTM2.csv, G02_LSTM1.csv etc.
# returns: dict of dicts => dict('G01': dict('LSTM1': [biases],
#                                            'LSTM2': [biases]),
#                                'G02': dict('LSTM1': [biases],
#                                            'LSTM2': [biases]) 
#                                                          ...)
def read_multiple_predictions(data_dir):
    sat_data = dict()
    sats_ommited = []
    for r, _, files in os.walk(data_dir):
        for filename in files:
            if '.csv' in filename:
                sat_and_pred = filename.split('.')[0]
                sat_name = sat_and_pred.split('_')[0]
                pred_name = sat_and_pred.split('_')[1]

                if sat_name in SATELLITES:
                    data = read_csv_data(os.path.join(r, filename))
                    if len(data) == 0 or len(data) != DEFAULT_DATA_LEN:
                        sats_ommited.append(sat_name)
                        if sat_name not in SATS_TO_OMMIT:
                            SATS_TO_OMMIT.append(sat_name)
                    else:
                        if sat_name not in sat_data:
                            sat_data[sat_name] = dict()
                        sat_data[sat_name][pred_name] = data

    ## Just for debug purposes!
    # print('Found {} of {} satallite data in {}'.format(\
    #     len(sat_data), len(SATELLITES), data_dir), end='')
    # print('') if len(sats_ommited) == 0 else print('. Omitted: {}'.format(\
    #     sats_ommited)) 

    return sat_data


def read_csv_data(directory):
    return pd.read_csv(directory, sep=';',header=0, parse_dates=[0], \
        index_col=0, squeeze=True).values


# returns dict('G01': [biases], 'G02': [biases] ...)
def read_data(data_dir):
    # os.walk() returns root, dirs and files
    sat_data = dict()
    sats_ommited = []
    for r, _, files in os.walk(data_dir):
        for filename in files:
            if '.csv' in filename:
                sat_name = filename.split('.')[0]
                if sat_name in SATELLITES:
                    data = read_csv_data(os.path.join(r, filename))
                    if len(data) == 0 or len(data) != DEFAULT_DATA_LEN:
                        sats_ommited.append(sat_name)
                        if sat_name not in SATS_TO_OMMIT:
                            SATS_TO_OMMIT.append(sat_name)
                    else:
                        sat_data[sat_name] = data

    ## Just for debug purposes!
    # print('Found {} of {} satallite data in {}'.format(\
    #     len(sat_data), len(SATELLITES), data_dir), end='')
    # print('') if len(sats_ommited) == 0 else print('. Omitted: {}'.format(\
    #     sats_ommited)) 
   
    return sat_data


def read_args(argv):
    error_choice = argv[1]
    error = error_choice if error_choice in ERROR_TYPES else ERROR_TYPES[0]
    output_dir = argv[2]
    reference_data_dir = argv[3]
    predicted_data_dir = argv[4]
    compare_data_dir = argv[5]

    return error, output_dir, reference_data_dir, predicted_data_dir, \
        compare_data_dir


def main(argv):
    error_type, output_dir, ref_data_dir, pred_data_dir, \
        comp_data_dir = read_args(argv)
    ref_data = read_data(ref_data_dir)
    pred_data = read_multiple_predictions(pred_data_dir)
    comp_data = read_data(comp_data_dir)
    errors = calculate_errors(error_type, ref_data, pred_data, comp_data)
    output_errors(errors, error_type, output_dir)


# calculate_errors.py
if __name__ == '__main__':
    main(sys.argv)
