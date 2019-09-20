import logging
import pandas as pd
import numpy as np
import scripts.research.lstm_utils as lstm_utils

TRAIN_FILE_NAME_PATTERN = "conversions/train_data/raw_csv/G{0:02d}.csv"



class Experiment:

    def __init__(self, satellite_number: int):
        self.satellite_number = satellite_number

    def run(self, save_plots=False):
        fname = TRAIN_FILE_NAME_PATTERN.format(self.satellite_number)
        logging.info('Executing experiment on file {}'.format(fname))
        train_data = pd.read_csv(fname,  sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True)
        raw_series = train_data.values
        lstm_utils.plot_raw_data(raw_series, save_plots)

        diff_values = lstm_utils.diff(raw_series)
        coeff = np.polyfit(range(len(diff_values)), diff_values, 1)
        logging.info("Zróżnicowane obserwacje wpasowano w prostą: y = {:.8f}x + {:.8f}".format(coeff[0], coeff[1]))
        lstm_utils.plot_differences(diff_values, save_plots)



def main():
    logging.basicConfig(format='<%(asctime)s> : %(message)s', level=logging.INFO)
    e = Experiment(1)
    e.run()


if __name__ == '__main__':
    main()
