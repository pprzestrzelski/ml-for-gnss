import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import tensorflow as tf
from keras import regularizers
from math import floor


def build_ll_model(input_size: int, input_shape: int, hidden_factor: int):
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


def build_ll_model_for_phase_4(input_size: int, input_shape: int, input_layer_dropout: float,
                               input_layer_recurrent_dropout: float, hidden_layer_dropout: float,
                               hidden_layer_recurrent_dropout: float, input_layer_regularization: float,
                               hidden_layer_regularization: float, optimizer: str):
    HIDDEN_FACTOR_FROM_PHASE_3 = 2
    hidden_size = int(input_size*HIDDEN_FACTOR_FROM_PHASE_3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(input_size,
                                   dropout=input_layer_dropout,
                                   recurrent_dropout=input_layer_recurrent_dropout,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(input_layer_regularization),
                                   stateful=False,
                                   input_shape=input_shape
                                   ))
    model.add(tf.keras.layers.LSTM(hidden_size,
                                   dropout=hidden_layer_dropout,
                                   recurrent_dropout=hidden_layer_recurrent_dropout,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(hidden_layer_regularization),
                                   stateful=False
                                   ))
    model.add(tf.keras.layers.Dense(1,
                                    activation='linear'
                                    ))
    model.compile(loss='mse', optimizer=optimizer)
    return model


def build_models_for_phase_2(input_size:int, input_shape: int):
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


def build_models_for_phase_3(input_size: int, input_shape: int)-> dict:
    models = {}
    models['LL05'] = build_ll_model(input_size, input_shape, 0.5)
    models['LL1'] = build_ll_model(input_size, input_shape, 1)
    models['LL2'] = build_ll_model(input_size, input_shape, 2)
    models['LL3'] = build_ll_model(input_size, input_shape, 3)
    models['LL4'] = build_ll_model(input_size, input_shape, 4)

    return models


def build_models_for_phase_4(input_size: int, input_shape: int)-> dict:
    models = {}
    models['low_dropouts'] = build_ll_model_for_phase_4(input_size, input_shape, 0.1, 0.01, 0.1, 0.01, 0.001, 0.001, 'rmsprop')
    models['high_dropouts'] = build_ll_model_for_phase_4(input_size, input_shape, 0.5, 0.1, 0.5, 0.1, 0.001, 0.001, 'rmsprop')
    return models


model_builders = { 2: build_models_for_phase_2, 3: build_models_for_phase_3, 4: build_models_for_phase_4}


def build_models(input_size: int, input_shape: int, phase: int)-> dict:
    try:
        return model_builders[phase](input_size, input_shape)
    except KeyError:
        print(f'Phase {phase} not supported by network builder.')
        return {}
