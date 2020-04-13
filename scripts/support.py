import numpy as np
import json
import math

class DataPrerocessing:

    def __init__(self, window_size=32, initial_bias=0.0, initial_epoch=0.0, final_epoch=0.0,
                 epoch_step=0.0, mean=0.0, scale=1.0, training_coefficent=0.3):
        self.initial_bias = initial_bias
        self.initial_epoch = initial_epoch
        self.final_epoch = final_epoch
        self.epoch_step = epoch_step
        self.mean = mean
        self.scale = scale
        self.window_size = window_size
        self.training_coefficent = training_coefficent

    def fit_transform(self, bias: np.ndarray, epochs: np.ndarray, lock_scaling:bool)-> np.ndarray:
        self.fit_epochs(epochs)
        transformed = self.fit_transform_bias(bias, lock_scaling)
        return transformed

    def fit_epochs(self, epochs: np.ndarray):
        self.initial_epoch = epochs[0]
        self.final_epoch = epochs[-1]
        self.epoch_step = np.min(np.diff(epochs))

    def fit_transform_bias(self, bias: np.ndarray, lock_scaling:bool)-> np.ndarray:
        self.initial_bias = bias[0]
        transformed = np.diff(bias)
        if not lock_scaling: self.mean = np.mean(transformed)
        transformed = transformed - self.mean
        # Ugly but works, unlike pervious solution
        if not lock_scaling: self.scale = 1.0/max(np.max(transformed), 0.0-np.min(transformed))
        transformed = transformed * self.scale
        return transformed

    def prepare_windowed_data(self, transformed: np.ndarray)-> np.ndarray:
        windows = []
        outputs = []
        step = 0
        while step + self.window_size < transformed.shape[0]:
            windows.append(transformed[step:step+self.window_size])
            outputs.append(transformed[step+self.window_size])
            step += 1
        return np.asarray(windows), np.asarray(outputs)

    def reverse_transform(self, bias: np.ndarray)-> np.ndarray:
        bias /= self.scale
        bias += self.mean
        np.insert(bias, 0, self.initial_bias)
        return np.cumsum(bias)

    def build_epochs_for_prediction(self, predictions: np.ndarray)-> np.ndarray:
        epochs = np.full(predictions.shape, self.epoch_step)
        np.insert(epochs, self.final_epoch, 0)
        return np.cumsum(epochs)

    def to_json(self, filename:str):
        with open(filename, 'w') as json_file:
            json.dump(vars(self), json_file)

    # FIXME: DUPLCATE OF prepare_widnowed_data
    def get_windowed_data(self, transformed: np.ndarray):
        windows = []
        outputs = []
        step = 0
        while step + self.window_size < transformed.shape[0]:
            windows.append(transformed[step:step+self.window_size])
            outputs.append(transformed[step+self.window_size])
            step += 1
        return np.asarray(windows), np.asarray(outputs)

    def split_training_and_validation(self, inputs: np.ndarray, outputs: np.ndarray):
        train_size = math.floor(len(inputs)*self.training_coefficent)
        x_train = inputs[:train_size]
        y_train = outputs[:train_size]
        x_test = inputs[train_size:]
        y_test = outputs[train_size:]
        return x_train, y_train, x_test, y_test


    @staticmethod
    def load_json(filename:str):
        config = None
        with open(filename, 'r') as json_file:
            config = json.load(json_file)
        return DataPrerocessing(**config)    
