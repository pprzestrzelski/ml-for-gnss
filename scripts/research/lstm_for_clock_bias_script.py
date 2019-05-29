from pandas import read_csv
import numpy as np
from math import floor, sqrt, fabs
import tensorflow as tf
from scripts.research.lstm_utils import diff, scale, create_lstm_dataset, \
    plot_lstm_loss, plot_raw_data, plot_differences, plot_scaled_values, plot_prediction, plot_prediction_error, Scaler
from keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error


# # Parameters
GPS_SVN = '05'
TRAIN_DATA_COEFF = 0.8
epochs = 50
# steps_per_epoch = 10
batch_size = 128
look_back = 12
window = look_back  # is equal just for tests!
prediction_depth = 96

train_file_name = "conversions/train_data/raw_csv/G{}.csv".format(GPS_SVN)          # train + evaluate
ref_file_name = "conversions/igu_observed/raw_csv/G{}.csv".format(GPS_SVN)          # ref data to test ANN
igu_pred_file_name = "conversions/igu_predicted/raw_csv/G{}.csv".format(GPS_SVN)    # comparable predictions by IGU-P


def create_network(train_data):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32,
                                   dropout=0.2,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False,
                                   input_shape=(None, train_data.shape[-1])
                                   ))
    # model.add(tf.keras.layers.LSTM(256,
    #                                dropout=0.3,
    #                                recurrent_dropout=0.3,
    #                                return_sequences=True,
    #                                activation='relu',
    #                                kernel_regularizer=regularizers.l2(0.001),
    #                                stateful=False
    #                                ))
    model.add(tf.keras.layers.LSTM(128,
                                   dropout=0.5,
                                   recurrent_dropout=0.5,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False
                                   ))
    model.add(tf.keras.layers.Dense(1,
                                    activation='linear'
                                    ))
    model.compile(loss='mse', optimizer='rmsprop')

    return model


# Based on https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
def predict_with_lstm(model, scaled_data, window_size, depth):
    predictions = []
    # init windowed_data arr with window_size samples from the scaled_data
    windowed_data = list(scaled_data[-window_size:])

    # predict!
    for _ in range(depth):
        predictioner = np.array(windowed_data)
        yhat = model.predict(predictioner.reshape(1, 1, window_size), verbose=0)

        # add to the memory
        predictions.append(yhat)

        # prepare window for next prediction with one new prediction
        windowed_data.append(yhat)
        windowed_data.pop(0)

    return predictions


series = read_csv(train_file_name, sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True)
raw_series = series.values
plot_raw_data(raw_series)

# # Let's make data stationary!
diff_values = diff(raw_series)
plot_differences(diff_values)

# # Normalize input data
# 1. Scale using MinMaxScaler
# scaler, scaled_diff_values = scale(np.array(diff_values))
# 2. or scale with mean and max absolute error (seems to be better)
scaler = Scaler()
scaled_diff_values = scaler.do_scale(diff_values)
plot_scaled_values(scaled_diff_values)

# # Create LSTM data sets: X and Y
scaled_diff_values = np.array(scaled_diff_values)
scaled_diff_values = scaled_diff_values.reshape(scaled_diff_values.shape[0], 1)
X, Y = create_lstm_dataset(scaled_diff_values, look_back)

# # Split data to the train and valuation datasets
tr_count = floor(len(X) * TRAIN_DATA_COEFF)
X_tr = X[:tr_count, :]
Y_tr = Y[:tr_count]
X_val = X[tr_count:, :]
Y_val = Y[tr_count:]

# Reshape required for LSTM layers
X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

nn = create_network(X_tr)
history = nn.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_val, Y_val), verbose=1, shuffle=False)

plot_lstm_loss(history)

# # Predict
predicted = predict_with_lstm(nn, scaled_diff_values, window, prediction_depth)
predicted = np.array(predicted).reshape(len(predicted), 1)

# # Reverse scale process
predicted = scaler.inverse_transform(predicted)

# # Invert differentiation
predicted_biases = []
raw = list(raw_series)
for j in range(len(predicted)):
    val = predicted[j] + raw[-1]
    predicted_biases.append(val)
    raw.append(val)

# # Summary, comparison, plots
ref_biases = read_csv(ref_file_name, sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True).values
igu_pred_biases = read_csv(igu_pred_file_name, sep=';', header=0, parse_dates=[0], index_col=0, squeeze=True).values

plot_prediction(ref_biases, predicted_biases, igu_pred_biases)

lstm_mae = mean_absolute_error(ref_biases, predicted_biases)
lstm_mse = mean_squared_error(ref_biases, predicted_biases)
igu_p_mae = mean_absolute_error(ref_biases, igu_pred_biases)
igu_p_mse = mean_squared_error(ref_biases, igu_pred_biases)

print()
print("LSTM prediction:  MAE = {:.4f} MSE = {:.4f} RMS = {:.4f}".format(lstm_mae, lstm_mse, sqrt(lstm_mse)))
print("IGU-P prediction: MAE = {:.4f} MSE = {:.4f} RMS = {:.4f}".format(igu_p_mae, igu_p_mse, sqrt(igu_p_mse)))

lstm_abs_error = []
igu_p_abs_error = []
for k in range(len(ref_biases)):
    lstm_abs_error.append(fabs(predicted_biases[k] - ref_biases[k]))
    igu_p_abs_error.append(fabs(igu_pred_biases[k] - ref_biases[k]))

plot_prediction_error(lstm_abs_error, igu_p_abs_error)
