#!/usr/bin/env python
# -*- coding: utf-8 -*-

# WYWOŁANIE
#         print("Usage: compare_lstm_to_others <KATALOG_Z_DANYMI> <NAZWA_KOLUMNY> <KATALOG_Z_SIECIAMI>"
#              "<ROZMIAR_WEJSCIA> <GLEBOKOSC_PREDYKCJI> <WSPOLCZYNNIK_SKALOWANIA> <KATALOG_WYJŚCIOWY>")


import sys
import os
import predict_with_lstm as predictor

class Sat:

    def __init__(self):
        self.data_file = ''
        self.topology_file = ''
        self.weights_file = ''
        self.output_file = ''

    def __str__(self):
        return '{} {} {} {}'.format(self.data_file, self.topology_file,
                                    self.weights_file, self.output_file)

def main(argv):
    data_path = argv[1]
    column_name = argv[2]
    network_path = argv[3]
    input_size = argv[4]
    prediction_depth = argv[5]
    scaling_factor = argv[6]
    output_folder = argv[7]

    satellites = {}

    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.csv' in file:
                name = file.split('.')[0]
                if name not in satellites:
                    sat = Sat()
                    sat.data_file =  os.path.join(r, file)
                    sat.topology_file =  os.path.join(network_path, ''.join([name, '_model.json']))
                    sat.weights_file =  os.path.join(network_path, ''.join([name, '_weights.h5']))
                    sat.output_file =  os.path.join(output_folder, ''.join([name, '_lstm.csv']))
                    satellites[name] = sat


    for name, sat in satellites.items():
        print('{} -> {}'.format(name, sat))
        predictor.main(['', sat.data_file, column_name, sat.topology_file, sat.weights_file, input_size,
                        prediction_depth, scaling_factor, name, sat.output_file])


if __name__ == '__main__':
    main(sys.argv)
