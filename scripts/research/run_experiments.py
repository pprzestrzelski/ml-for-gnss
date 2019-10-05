import logging
import sys
import pandas as pd
import numpy as np
import scripts.research.lstm_utils as lstm_utils
import matplotlib.pyplot as plt
from scripts.core.ml.LinearEstimator import LinearEstimator
from scripts.core.ml.LSTMEstimator import LSTMEstimatorFactory
from sklearn.model_selection import train_test_split

TRAIN_FILE_NAME_PATTERN = 'conversions/train_data/raw_csv/G{0:02d}.csv'
REFERENCE_FILE_NAME_PATTERN = 'conversions/igu_observed/raw_csv/G{0:02d}.csv'
IGU_FILE_NAME_PATTERN = 'conversions/igu_predicted/raw_csv/G{0:02d}.csv'


class Experiment:

    def __init__(self, satellite_number: int):
        self.satellite_number = satellite_number
        self.sat_id = '{0:02d}'.format(self.satellite_number)
        self.predictions = {}

    def run(self, save_plots=False):
        self.__prepare_data()
        lstm_utils.plot_raw_data(self.raw_data, save_plots)
        diff_values = self.__differentiate_train_data()
        self.__plot_differences(diff_values, save_plots)
        self.__prepare_estimators()
        self.__run_estimators()
        self.__plot_predictions()

    def __prepare_data(self):
        logging.info('Executing experiment for satellite {}'.format(self.satellite_number))

        self.raw_data = pd.read_csv(TRAIN_FILE_NAME_PATTERN.format(self.satellite_number),
                                    sep=';', header=0).values
        self.__split_raw_into_train_and_test()
        self.reference_data = pd.read_csv(REFERENCE_FILE_NAME_PATTERN.format(self.satellite_number),
                                          sep=';', header=0, parse_dates=[0], index_col=0,
                                          squeeze=True).values
        self.igu_data = pd.read_csv(IGU_FILE_NAME_PATTERN.format(self.satellite_number),
                                    sep=';', header=0, parse_dates=[0], index_col=0,
                                    squeeze=True).values

    def __split_raw_into_train_and_test(self, test_size=0.5):
        epochs = []
        clock_biases = []
        for epoch, clock_bias in self.raw_data:
            epochs.append(epoch)
            clock_biases.append(float(clock_bias) * 10.0 ** 3)

        x = np.array(epochs).reshape(-1, 1)
        y = np.array(clock_biases)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=test_size,
                                                                                shuffle=False)

    def __differentiate_train_data(self):
        diff_values = lstm_utils.diff(self.y_train)
        coeff = np.polyfit(range(len(diff_values)), diff_values, 1)
        logging.info("Differentiation is approximadet by linear equation: y = {:.8f}x + {:.8f}".format(coeff[0], coeff[1]))
        return diff_values

    def __prepare_estimators(self):
        self.estimators = \
            {'Linear': LinearEstimator(self.x_train, self.x_test, self.y_train, self.y_test, self.sat_id),
             'Poly-2nd': LinearEstimator(self.x_train, self.x_test, self.y_train, self.y_test, self.sat_id,
                                         estimator='OLS', degree=2),
             'Poly-4th': LinearEstimator(self.x_train, self.x_test, self.y_train, self.y_test, self.sat_id,
                                         estimator='OLS', degree=4),
             'Poly-8th': LinearEstimator(self.x_train, self.x_test, self.y_train, self.y_test, self.sat_id,
                                         estimator='OLS', degree=8),
             'LSTM': LSTMEstimatorFactory().build_double_layer_estimator(self.x_train, self.x_test, self.y_train,
                                                                         self.y_test, self.sat_id,3)
             }

    def __run_estimators(self):
        for name, estimator in self.estimators.items():
            estimator.fit()
            estimator.predict()
            res = estimator.stats()
            mae, mse, rms = res
            self.predictions[name] = estimator.y_pred
            logging.info('{0:s} => MAE = {1:2.4f} MSE = {2:2.4f} RMS = {3:2.4f}'.format(name, mae, mse, rms))

    def __plot_differences(self, data, print_plot=False):
        plt.plot(data, 'k')
        epochs = len(data)
        logging.info("Plot differences")
        logging.info("Max diff: {}".format(max(data)))
        logging.info("Min diff: {}".format(min(data)))
        plt.xlabel('Epoka')
        plt.ylabel('Różnica opóźnień [ns]')
        #plt.xticks(np.arange(0, epochs, 192))
        #plt.yticks(np.arange(-3, 3.01, 0.5))
        #plt.xlim([0, epochs])
        #plt.ylim([-3, 3])
        plt.show()

    def __plot_predictions(self):
        for name, prediction in self.predictions.items():
            plt.plot(prediction, label=name)
            #plt.yticks(np.arange(-3, 3.01, 0.5))
            #plt.ylim([-3, 3])
            plt.legend()
        plt.show()


def main():
    logging.basicConfig(format='<%(asctime)s> : %(message)s', level=logging.INFO)
    e = Experiment(1)
    e.run()


if __name__ == '__main__':
    main()
