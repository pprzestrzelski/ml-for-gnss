from sklearn.linear_model import \
    (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, SGDRegressor, Ridge)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# https://scikit-learn.org/stable/auto_examples/linear_model/
# plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
class LinearEstimator:
    """
    Linear estimators dedicated to GNSS satellite data prediction.
    Therefore some descriptions may be domain specific, e.g. plot title.
    """
    def __init__(self, x_train, x_test, y_train, y_test, sat_name, estimator="OLS", degree=1):
        self.estimator_name = None
        self.regressor = None
        self.degree = None
        self.satellite_name = sat_name
        self.estimators = \
            {'OLS': LinearRegression(),
             'Theil-Sen': TheilSenRegressor(random_state=42),
             'RANSAC': RANSACRegressor(random_state=42),
             'HuberRegressor': HuberRegressor(),
             'SGD': SGDRegressor(random_state=42),
             'Ridge': Ridge(random_state=42)}
        self.set_estimator(estimator, degree)
        self.data_trained = False
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.df = None
        self.mae = None     # Mean Absolute Error
        self.mse = None     # Mean Square Error
        self.rms = None     # Root Mean Square Error

    def set_estimator(self, estimator, degree=1):
        if estimator not in self.estimators:
            print("ERROR: {} is not available (tip: use available_estimators())".format(estimator))
        else:
            self.estimator_name = estimator
            self.degree = degree
            self.regressor = make_pipeline(PolynomialFeatures(self.degree), self.estimators[estimator])
            self.__clear_all()

    def fit(self):
        if self.estimator_name:
            self.regressor.fit(self.x_train, self.y_train)
            self.data_trained = True
        else:
            print("ERROR: object was not initialized correctly!")

    def predict(self):
        if self.data_trained:
            self.y_pred = self.regressor.predict(self.x_test)
            self.__calculate_statistics()
            self.__create_pandas_data_frame()
        else:
            print("ERROR: could not make prediction, train data first!")

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

    def print_stats(self):
        print('Liner Regression R squared: %.4f' % self.regressor.score(self.x_test, self.y_test))
        print('Mean Absolute Error: {}'.format(self.mae))
        print('Mean Squared Error: {}'.format(self.mse))
        print('Root Mean Squared Error: {}'.format(self.rms))

    def stats(self):
        return [self.mae, self.mse, self.rms]

    def pandas_data_frame(self):
        return self.df

    def available_estimators(self):
        out = []
        for name in self.estimators.keys():
            out.append(name)
        return out

    def __clear_all(self):
        self.data_trained = False
        self.y_pred = None
        self.df = None
        self.mae = None
        self.mse = None
        self.rms = None

    def __calculate_statistics(self):
        self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.mse = metrics.mean_squared_error(self.y_test, self.y_pred)
        self.rms = np.sqrt(self.mse)

    def __create_pandas_data_frame(self):
        self.df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
