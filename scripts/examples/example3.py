import numpy as np
from sklearn.model_selection import train_test_split
from core.gnss.gnss_clock_data import GnssClockData
from core.ml.SVREstimator import SVREstimator

SCALE = 10.0 ** 9


def main():
    clock_data = GnssClockData(dir_name="clock_data")
    sat_number = 'G05'
    data = clock_data.get_satellite_data(sat_number)
    epochs = []
    clock_biases = []
    for epoch, clock_bias in data:
        epochs.append(epoch)
        clock_biases.append(float(clock_bias.get_bias()) * SCALE)

    X = np.array(epochs).reshape(-1, 1)
    y = np.array(clock_biases)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    regressor = SVREstimator(X_train, X_test, y_train, y_test, sat_number, gamma=0.2, epsilon=0.1)
    regressor.fit()
    regressor.predict()
    regressor.print_stats()
    regressor.plot_prediction()
    regressor.plot_prediction_error()


if __name__ == '__main__':
    main()
