import os
import logging
import tensorflow as tf
import numpy as np
from scripts.core.ml.Estimator import Estimator
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers
from scripts.research.lstm_utils import Scaler 

# https://keras.io/preprocessing/sequence/
# https://keras.io/models/model/
# https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
class LSTMEstimator(Estimator):

    LSTM_MODEL_DIR = "lstm_models"

    def __init__(self, x_train, x_test, y_train, y_test, sat_name, model, scaler,
                 window_size):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.loss = None
        self.history = None
        self.regressor = model
        self.scaler = scaler
        self.window_size = window_size
        self.__transdorm_data_for_lstm()

    def build_model(self):
        if self.regressor is None:
            self.regressor = Sequential()
        else:
            logging.warning('Model was already build.')

    def save_model(self, name):
        self.regressor.save(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')


    def load_model(self, name):
        self.regressor = load_model(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')

    def fit(self):
        self.history = self.regressor.fit(
            self.x_train, self.y_train,
            epochs=1000, validation_data=(self.x_test, self.y_test),
            shuffle=False, verbose=self.verbose_level)

    def evaluate(self):
        self.loss = self.regressor.evaluate(self.x_train, self.y_train, verbose=self.verbose_level)

    def predict(self):
        self.y_pred = self.regressor.predict(self.x_test, verbose=self.verbose_level)
        self.__calculate_statistics()
        # self.__create_pandas_data_frame()

    def get_fit_history(self):
        return self.history

    def calculate_fitness(self):
        self.fitness = self.loss

    def __transdorm_data_for_lstm(self):
        
        self.x_train, self.y_train = self.__create_lstm_dataset(self.y_train)
        self.x_test, self.y_test = self.__create_lstm_dataset(self.y_test)

    def __create_lstm_dataset(self, dataset):
        data_x, data_y = [], []
        for i in range(len(dataset) - self.window_size - 1):
            a = dataset[i: (i + self.window_size)]
            data_x.append(a)
            data_y.append(dataset[i + self.window_size])
        return np.array(data_x), np.array(data_y)


class LSTMEstimatorFactory:

    def __init__(self):
        pass

    def build_double_layer_estimator(self, x_train, x_test, y_train, y_test, sat_name, window_size, scale=0):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(32,
                                       dropout=0.2,
                                       recurrent_dropout=0.2,
                                       return_sequences=True,
                                       activation='relu',
                                       kernel_regularizer=regularizers.l2(0.001),
                                       stateful=False,
                                       input_shape=(None, window_size)
                                       ))
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

        scaler = None
        if scale != 0:
            logging.error('Custom scaling not yet implemented.')
        
        estimator = LSTMEstimator(x_train, x_test, y_train, y_test, sat_name, model, scaler,
                                  window_size)

        return model
