#!/usr/bin/env python
# -*- coding: utf-8 -*-

# WYWOŁANIE
#         print("Usage: compare_lstm_to_others <KATALOG_Z_DANYMI> <NAZWA_KOLUMNY> <KATALOG_Z_SIECIAMI>"
#              "<ROZMIAR_WEJSCIA> <GLEBOKOSC_PREDYKCJI> <WSPOLCZYNNIK_SKALOWANIA> <KATALOG_WYJŚCIOWY>")


import sys
import os
import predict_with_lstm as predictor


def main(argv):
  
    column_name = argv[1]
    input_dir = argv[2]
    model_dir = argv[3]
    input_size = int(argv[4])
    prediction_depth = int(argv[5])
    scale = float(argv[6])
    output_dir = argv[7]
    last_epoch = float(argv[8])
    epoch_step = float(argv[9])


    satellites = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_dir):
        for file in f:
            if '.csv' in file:
                name = file.split('.')[0]
                if name not in satellites:
                    satellites.append(name)


    for sat_name in satellites:
        predictor.main(['', sat_name, column_name, input_dir, model_dir, input_size,
                        prediction_depth, scale, output_dir, last_epoch, epoch_step])



if __name__ == '__main__':
    main(sys.argv)
