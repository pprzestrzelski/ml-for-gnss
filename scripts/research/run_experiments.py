import logging
import pandas as pd
import numpy as np
import scripts.research.lstm_utils as lstm_utils
from scripts.core.ml.LinearEstimator import LinearEstimator

TRAIN_FILE_NAME_PATTERN = 'conversions/train_data/raw_csv/G{0:02d}.csv'
REFERENCE_FILE_NAME_PATTERN = 'conversions/igu_observed/raw_csv/G{0:02d}.csv'
IGU_FILE_NAME_PATTERN = 'conversions/igu_predicted/raw_csv/G{0:02d}.csv'


class Experiment:

    def __init__(self, satellite_number: int):
        self.satellite_number = satellite_number

    def run(self, save_plots=False):
        self.__prepare_data()
        lstm_utils.plot_raw_data(self.train_data, save_plots)
        diff_values = self.__differentiate_train_data()
        lstm_utils.plot_differences(diff_values, save_plots)

    def __prepare_data(self):
        logging.info('Executing experiment for satellite {}'.format(self.satellite_number))

        self.train_data = pd.read_csv(TRAIN_FILE_NAME_PATTERN.format(self.satellite_number),
                                      sep=';', header=0, parse_dates=[0], index_col=0,
                                      squeeze=True).values
        self.reference_data = pd.read_csv(REFERENCE_FILE_NAME_PATTERN.format(self.satellite_number),
                                          sep=';', header=0, parse_dates=[0], index_col=0,
                                          squeeze=True).values
        self.igu_data = pd.read_csv(IGU_FILE_NAME_PATTERN.format(self.satellite_number),
                                    sep=';', header=0, parse_dates=[0], index_col=0,
                                    squeeze=True).values

    def __differentiate_train_data(self):
        diff_values = lstm_utils.diff(self.train_data)
        coeff = np.polyfit(range(len(diff_values)), diff_values, 1)
        logging.info("Differentiation is approximadet by linear equation: y = {:.8f}x + {:.8f}".format(coeff[0], coeff[1]))
        return diff_values

    def __prepare_estimators(self):
        self.estimators = \
            {'Linear': LinearEstimator(X_train, X_test, y_train, y_test, GPS_SVN),
             'Poly-2nd': LinearEstimator(X_train, X_test, y_train, y_test, GPS_SVN, estimator='OLS', degree=2),
             'Poly-4th': LinearEstimator(X_train, X_test, y_train, y_test, GPS_SVN, estimator='OLS', degree=4),
             'Poly-8th': LinearEstimator(X_train, X_test, y_train, y_test, GPS_SVN, estimator='OLS', degree=8)}

def main():
    logging.basicConfig(format='<%(asctime)s> : %(message)s', level=logging.INFO)
    e = Experiment(1)
    e.run()


if __name__ == '__main__':
    main()
