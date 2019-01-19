from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use GridSearchCV (https://scikit-learn.org/stable/modules/generated/
# sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
# to find the most optimal SVR parameters!
# example: https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
# More to read: https://scikit-learn.org/stable/modules/grid_search.html#grid-search (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)


class SVREstimator:
    def __init__(self, x_train, x_test, y_train, y_test, sat_name,
                 kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1, degree=3):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.satellite_name = sat_name
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, degree=degree)
        self.y_pred = None
        self.df = None
        self.mae = None     # Mean Absolute Error
        self.mse = None     # Mean Square Error
        self.rms = None     # Root Mean Square Error

    def fit(self):
        self.regressor.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_pred = self.regressor.predict(self.x_test)
        self.__calculate_statistics()
        self.__create_pandas_data_frame()

    def prediction(self):
        return self.y_pred

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
        print('Liner Regression R squared: %.4f' % self.regressor.score(self.x_test, self.y_test))
        print('Mean Absolute Error: {}'.format(self.mae))
        print('Mean Squared Error: {}'.format(self.mse))
        print('Root Mean Squared Error: {}'.format(self.rms))

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
