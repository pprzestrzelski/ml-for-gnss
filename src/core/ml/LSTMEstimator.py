import os
from src.core.ml.Estimator import Estimator
from keras.models import Sequential
from keras.models import load_model


# https://keras.io/preprocessing/sequence/
# https://keras.io/models/model/
# https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
class LSTMEstimator(Estimator):

    LSTM_MODEL_DIR = "lstm_models"

    def __init__(self, x_train, x_test, y_train, y_test, sat_name, verbose_level=0):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.loss = None
        self.history = None
        self.verbose_level = verbose_level

    def build_model(self):
        if self.regressor is None:
            self.regressor = Sequential()
        else:
            print("WARNING: model was already build.")

    def rebuild_model(self):
        self.regressor = Sequential()

    def save_model(self, name):
        self.regressor.save(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')

    def load_model(self, name):
        self.regressor = load_model(LSTMEstimator.LSTM_MODEL_DIR + os.sep + name + '.h5')

    def add(self, layer):
        if self.regressor is not None:
            self.regressor.add(layer)
        else:
            print("ERROR: build model first!")

    def compile(self, loss='mean_squared_error', optimizer='adam'):
        self.regressor.compile(loss=loss, optimizer=optimizer)

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
