#!/usr/bin/python3

import sys
import os
import compare_lstm_to_others as comparator

satellites = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31',  'G32']

def main(argv):
    lstm_directory = argv[1]
    igu_directory = argv[2]
    reference_directory = argv[3]
    output_directory = argv[4]

        # r=root, d=directories, f = files
    for r, d, f in os.walk(reference_directory):
        for filename in f:
            if '.csv' in filename:
                name = filename.split('.')[0]
                print(">> Results for {} satellite".format(name))
                try:
                    if name in satellites:
                        ref_file =  os.path.join(r, filename)
                        lstm_file =  os.path.join(lstm_directory, ''.join([name, '_lstm.csv']))
                        igu_file = os.path.join(igu_directory, ''.join([name, '.csv']))
                        output_csv = os.path.join(output_directory, ''.join([name, '.csv']))
                        output_image = os.path.join(output_directory, ''.join([name, '.png']))
                        comparator.main(['', lstm_file, igu_file, ref_file, output_csv, output_image])
                except FileNotFoundError as fnfe:
                    print("Missing data for satellite {}".format(name))

if __name__ == '__main__':
    main(sys.argv)
