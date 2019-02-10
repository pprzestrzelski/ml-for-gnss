from abc import ABC, abstractmethod
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Estimator(ABC):
    def __init__(self, x_train, x_test, y_train, y_test, sat_name):
        ABC.__init__(self)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.satellite_name = sat_name
        self.y_pred = None
        self.df = None
        self.mae = None     # Mean Absolute Error
        self.mse = None     # Mean Square Error
        self.rms = None     # Root Mean Square Error
        self.fitness = None     # Liner Regression R squared or Losses function result for NN
        self.regressor = None

    def fit(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_pred = self.regressor.predict(self.x_test)
        self.__calculate_statistics()
        self.calculate_fitness()
        self.__create_pandas_data_frame()

    def prediction(self):
        return self.y_pred

    @abstractmethod
    def calculate_fitness(self):
        pass

    def plot_prediction(self):
        plt.plot([i for i in range(len(self.y_test))], self.y_test, 'o', label="Real value")
        plt.plot([i for i in range(len(self.y_pred))], self.y_pred, 'x', label="Prediction")
        plt.title('Satellite {} clock corrections and their predictions'.format(self.satellite_name))
        plt.ylabel('Clock correction [ns]')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def plot_prediction_error(self):
        diff = self.y_test - self.y_pred
        plt.plot([i for i in range(len(diff))], diff, '-r', label="Prediction")
        plt.title('Clock correction prediction error ({})'.format(self.satellite_name))
        plt.ylabel('Error [ns]')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def print_stats(self):
        print('Fitness parameter: {0:.4f}'.format(self.fitness))
        print('Mean Absolute Error: {0:.4f}'.format(self.mae))
        print('Mean Squared Error: {0:.4f}'.format(self.mse))
        print('Root Mean Squared Error: {0:.4f}'.format(self.rms))

    def stats(self):
        return [self.mae, self.mse, self.rms]

    def pandas_data_frame(self):
        return self.df

    def __calculate_statistics(self):
        self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.mse = metrics.mean_squared_error(self.y_test, self.y_pred)
        self.rms = np.sqrt(self.mse)

    def __create_pandas_data_frame(self):
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
