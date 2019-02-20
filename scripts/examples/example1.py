# DESCRIPTION:
# File presents comparison of LinearRegression (from scikit-learn package) and LinearEstimator.py

from core.gnss.gnss_clock_data import GnssClockData
from core.ml.LinearEstimator import LinearEstimator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

SCALE = 10.0 ** 9   # ns    1 ns => ~30 cm of an error in pseudorange


def main():
    # Get GNSS clock satellite data - reads from clock_data directory in default
    clock_data = GnssClockData(dir_name="clock_data")

    # Choose satellite to investigate
    # (interesting data: G05, G23, G24)
    sat_number = 'G05'

    # Split data (epoch_data, clock_data) manually
    data = clock_data.get_satellite_data(sat_number)
    epochs = []
    clock_biases = []   # clock data is given in seconds, we should scale this data (best to ns)

    for epoch, clock_bias in data:
        epochs.append(epoch)
        clock_biases.append(float(clock_bias.get_bias()) * SCALE)

    X = np.array(epochs).reshape(-1, 1)
    y = np.array(clock_biases)

    print("\n=== Plot all satellite data")
    plt.plot([i for i in range(len(y))], y, '-b')
    plt.title('Satellite {} clock corrections'.format(sat_number))
    plt.ylabel('Clock correction [ns]')
    plt.xlabel('Epoch')
    plt.show()

    print("\n=== Start prediction using {} for satellite {}".format("linear regression", sat_number))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
    print("Train data: {}, test data: {}".format(len(X_train), len(X_test)))

    print("\n=== Linear estimation using LinearEstimator.py:")
    est = LinearEstimator(X_train, X_test, y_train, y_test, sat_number)
    est.fit()
    est.predict()
    est.print_stats()
    est.plot_prediction()

    print("\n=== Linear estimation using LinearRegression (scikit-learn package):")
    regressor = LinearRegression().fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print('Liner Regression R squared: %.4f' % regressor.score(X_test, y_test))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    plt.plot([i for i in range(len(y_test))], y_test, 'o', label="Real value")
    plt.plot([i for i in range(len(y_pred))], y_pred, 'x', label="Prediction")
    plt.title('Satellite {} clock corrections and their predictions'.format(sat_number))
    plt.ylabel('Clock correction [ns]')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    n = 5
    print("\n=== Show first {} predictions".format(n))
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df.head(n))

    diff = y_test - y_pred
    plt.plot([i for i in range(len(diff))], diff, '-r', label="Prediction")
    plt.title('Clock correction prediction error ({})'.format(sat_number))
    plt.ylabel('Error [ns]')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
