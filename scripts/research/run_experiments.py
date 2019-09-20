import logging
import pandas as pd

TRAIN_FILE_NAME_PATTERN = "conversions/train_data/raw_csv/G{0:02d}.csv"



class Experiment:

    def __init__(self, satellite_number: int):
        self.satellite_number = satellite_number

    def run(self):
        fname = TRAIN_FILE_NAME_PATTERN.format(self.satellite_number)
        logging.info('Executing experiment on file {}'.format(fname))
        train_data = pd.read_csv(fname,  sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True)



def main():
    logging.basicConfig(format='<%(asctime)s> : %(message)s', level=logging.INFO)
    e = Experiment(1)
    e.run()


if __name__ == '__main__':
    main()
