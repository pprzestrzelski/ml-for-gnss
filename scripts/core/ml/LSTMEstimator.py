import os
import logging
import tensorflow as tf
from scripts.core.ml.Estimator import Estimator
from keras.models import Sequential
from keras.models import load_model


# https://keras.io/preprocessing/sequence/
# https://keras.io/models/model/
# https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
class LSTMEstimator(Estimator):

    LSTM_MODEL_DIR = "lstm_models"

    def __init__(self, x_train, x_test, y_train, y_test, sat_name, model):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.loss = None
        self.history = None
        self.regressor = model

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


class LSTMEstimatorFactory:

    def __init__(self):
        pass

    def build_double_layer_estimator(self):
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