#!/usr/bin/python3

import sys
import compare_lstm_to_others as comparator


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
                if name not in satellites:
                    ref_file =  os.path.join(r, filename)
                    lstm_file =  os.path.join(lstm_directory, ''.join([name, '.csv']))
                    igu_file = os.path.join(lstm_directory, ''.join([name, '.csv']))
                    output_csv = os.path.join(output_directory, ''.join([name, '.csv']))
                    output_image = os.path.join(output_directory, ''.join([name, '.png']))
                    comparator.main(['',lstm_file, igu_file, ref_file, output_csv, output_image])


if '__name__' == '__main__':
    main(sys.argv)
