import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import tensorflow as tf
from keras import regularizers
from math import floor




def build_ll_model(input_size:int, input_shape: int, hidden_factor: int):
    hidden_size = int(input_size*hidden_factor)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(input_size,
                                   dropout=0.2,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False,
                                   input_shape=input_shape
                                   ))
    model.add(tf.keras.layers.LSTM(hidden_size,
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

def build_dl_model(input_size:int, hidden_factor: int):
    hidden_size = int(input_size*hidden_factor)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_size, activation='relu'))
    model.add(tf.keras.layers.LSTM(hidden_size,
                                   dropout=0.5,
                                   recurrent_dropout=0.5,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(0.001),
                                   stateful=False
                                   ))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model



def build_models(input_size:int, input_shape: int):
    models = {}
    models['LL05'] = build_ll_model(input_size, input_shape, 0.5)
    models['LL1'] = build_ll_model(input_size, input_shape, 1)
    models['LL2'] = build_ll_model(input_size, input_shape, 2)
    models['LL3'] = build_ll_model(input_size, input_shape, 3)
    models['LL4'] = build_ll_model(input_size, input_shape, 4)

    models['DL05'] = build_dl_model(input_size, 0.5)
    models['DL1'] = build_dl_model(input_size, 1)
    models['DL2'] = build_dl_model(input_size, 2)
    models['DL3'] = build_dl_model(input_size, 3)
    models['DL4'] = build_dl_model(input_size, 4)

    return models
