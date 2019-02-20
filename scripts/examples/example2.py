# DESCRIPTION
# Simple demonstration of various estimators available in LinearEstimator.py

import numpy as np
from sklearn.model_selection import train_test_split
from core.gnss.gnss_clock_data import GnssClockData
from core.ml.LinearEstimator import LinearEstimator

SCALE = 10.0 ** 9


def main():
    clock_data = GnssClockData(dir_name="clock_data")
    sat_number = 'G09'
    data = clock_data.get_satellite_data(sat_number)
    epochs = []
    clock_biases = []
    for epoch, clock_bias in data:
        epochs.append(epoch)
        clock_biases.append(float(clock_bias.bias) * SCALE)

    X = np.array(epochs).reshape(-1, 1)
    y = np.array(clock_biases)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    estimators = \
        {'Linear': LinearEstimator(X_train, X_test, y_train, y_test, sat_number),
         'Poly-2nd': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='OLS', degree=2),
         'Poly-4th': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='OLS', degree=4),
         'Poly-8th': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='OLS', degree=8),
         'Theil-Sen': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='Theil-Sen'),
         'RANSAC': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='RANSAC'),
         'HuberRegressor': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='HuberRegressor'),
         'SGD': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='SGD'),
         'Ridge': LinearEstimator(X_train, X_test, y_train, y_test, sat_number, estimator='Ridge')}

    for name, est in estimators.items():
        est.fit()
        est.predict()
        res = est.stats()
        mae, mse, rms = res
        print("%15s => mae: %.4f, mse: %.4f, rms: %.4f" % (name, mae, mse, rms))


if __name__ == '__main__':
    main()
