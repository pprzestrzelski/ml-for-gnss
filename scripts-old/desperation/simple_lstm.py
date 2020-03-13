#!/usr/bin/env python

# Importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Wczytujemy dane plik z danymi podany jako pierwszy argument w terminalu
dataset = pd.read_csv(sys.argv[1], sep=';')

# Wyciągamy interesujące nas dane z zbioru danych, nazwa kolumny jest podana
# jako drugi parametr
time_series = dataset[sys.argv[2]].to_numpy()

# Skalujemy dane tak żeby nie wychodziły poza przedział <-1;1>
scaler = MinMaxScaler(feature_range=(-1,1))
time_series_for_scaler = time_series.reshape(1, -1) # Fit musi mieć tablicę 2D
scaled_time_series = scaler.fit_transform(time_series_for_scaler)
scaled_time_series = scaled_time_series.reshape(-1) # I z powrotem do jednowymiarowej

# Generujemy wejścia i wyjścia dla sieci

INPUT_SIZE = 30
inputs = []
outputs = []
print('Size of scaled time series = {}'.format(scaled_time_series.shape))
for i in range(INPUT_SIZE, scaled_time_series.shape[0]):
    inputs.append(scaled_time_series[i-INPUT_SIZE:i])
    outputs.append(scaled_time_series[i])

# Przekształcamy do tablic numpy
inputs = np.asarray(inputs)
outputs = np.asarray(outputs)

print('Input shape : {}'.format(inputs.shape))
print('Output shape : {}'.format(outputs.shape))

# Przekształcamy wejścia tak żeby pasowały do sieci LSTM
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))


# Tworzymy sieć
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(inputs.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1)) # Warstwa wyjściowa

# Kompilacja sieci
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Uczenie dla danych wejściowych i wyjściowych
model.fit(inputs, outputs, epochs = 100, batch_size = 32)
