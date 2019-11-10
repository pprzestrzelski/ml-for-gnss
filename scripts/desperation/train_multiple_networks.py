#!/usr/bin/env python

# WYWOŁANIE
# train_multiple_networks <KATALOG_Z_DANYMI> <NAZWA_KOLUMNY> <ROZMIAR_WEJŚCIA_SIECI> <ILOŚĆ_EPOK>
# <STOSUNEK TRENING/TEST> <KATALOG_Z_REZULTATAMI>


import sys
import os
import scripts.desperation.train_for_gnss as training_script


def main(argv):
    path = argv[1]
    files = []
    names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
                names.append(file.split('.')[0])

    for i in range(len(files)):
        print('{} -> {}'.format(names[i], files[i]))
        training_script.main(['', files[i], argv[2], argv[3], argv[4], argv[5], names[i], argv[6]])


if __name__ == '__main__':
    main(sys.argv)
