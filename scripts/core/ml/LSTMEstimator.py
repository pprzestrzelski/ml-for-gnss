import os
import logging
import tensorflow as tf
import numpy as np
from scripts.core.ml.Estimator import Estimator
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers
from scripts.research.lstm_utils import Scaler 


class LstmDataShape:

    def __init__(self, batch: int, steps: int, window: int):
        self.batch = batch
        self.steps = steps
        self.window = window

# https://keras.io/preprocessing/sequence/
# https://keras.io/models/model/
# https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
class LSTMEstimator(Estimator):

    LSTM_MODEL_DIR = "lstm_models"

    def __init__(self, x_train: np.ndarray, x_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 sat_name: str, model: Sequential, scaler: Scaler,
                 data_shape: LstmDataShape):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.loss = None  # FIXME: ADD TYPE HINT
        self.history = None  # FIXME: ADD TYPE HINT
        self.model = model
        self.scaler = scaler
        self.data_shape = data_shape
        self.__transform_data_for_lstm()

    def build_model(self):
        if self.model is None:
            self.model = Sequential()
        else:
            logging.warning('Model was already build.')

    def save_model(self, name):
        self.model.save(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')


    def load_model(self, name):
        self.model = load_model(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')

    def fit(self):
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=1000, validation_data=(self.x_test, self.y_test),
            shuffle=False, steps_per_epoch=10,
            validation_steps=3)

    def evaluate(self):
        self.loss = self.model.evaluate(self.x_train, self.y_train, verbose=self.verbose_level)

    def predict(self):
        self.y_pred = self.model.predict(self.x_test, verbose=self.verbose_level)
        self.__calculate_statistics()
        # self.__create_pandas_data_frame()

    def get_fit_history(self):
        return self.history

    def calculate_fitness(self):
        self.fitness = self.loss

    def __transform_data_for_lstm(self):
        
        self.x_train, self.y_train = self.__create_lstm_dataset(self.y_train)
        self.x_test, self.y_test = self.__create_lstm_dataset(self.y_test)

    def __create_lstm_dataset(self, input_sequence: list) -> (np.ndarray, np.ndarray):
        data_x = self.__sequence_to_numpy_windows(input_sequence)
        data_y = self.__sequence_to_numpy_window_responses(input_sequence)
        return data_x, data_y

    def __sequence_to_numpy_windows(self, sequence: list) -> np.ndarray:
        windows = []
        for i in range(len(sequence) - self.data_shape.window - 1):
            windows.append(sequence[i:(i+self.data_shape.window)])
        return np.array(windows)

    def __sequence_to_numpy_window_responses(self, sequence: list) -> np.ndarray:
        responses = []
        for i in range(len(sequence) - self.data_shape.window - 1):
            responses.append(sequence[i + self.data_shape.window])
        return np.array(responses)

    def __split_sequence_into_batches(self, sequence: np.ndarray) -> (np.ndarray, int):
        batch_size = sequence.shape[0] // self.data_shape.window*self.data_shape.steps
        lost_elements = sequence.shape[0] - batch_size*self.data_shape

    def __prepare_


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

        return estimator
