#!/usr/bin/env python

import sys
import os
import pandas as pd
import numpy as np

# get_scaling_factor <KATALOG_Z_DANYMI> <NAZWA_KOLUMNY>

def diff(dataset):
    diffs = list()
    for i in range(1, len(dataset)):
        diffs.append(dataset[i] - dataset[i - 1])
    return np.asarray(diffs)


def main(argv):
    path = argv[1]
    max_val = 0.0
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                filename = os.path.join(r, file)
                #print('FNAME = {}'.format(filename))
                dataset = pd.read_csv(filename, sep=';')
                time_series = dataset[argv[2]].to_numpy()
                time_series = diff(time_series)
                max_val = max(max_val, max(time_series.max(), abs(time_series.min())))
    scaling_factor = 1.0/max_val
    print('Maximum encountered value is = {}'.format(max_val))
    print('Suggested scaling factor is = {}'.format(scaling_factor))


if __name__ == '__main__':
    main(sys.argv)