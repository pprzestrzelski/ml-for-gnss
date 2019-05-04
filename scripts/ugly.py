import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from math import floor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Podajemy źródło danych jako pierwszy argument
data = pd.read_csv(sys.argv[1], sep=';')

t0 = data['Epoch'][0] # Wartość pierwszej epoki
dt = data['Epoch'][1] - data['Epoch'][0] # Różnice pomiędzy epokami
e0 = data['Clock_bias'][0] # Bias w zerowym kroku

# Bierzemy tylko bias i przekształcamy go na różnice pomiędzy kolejnymi
# wartościami. Oczywiście taki ciąg nie zawiera odpowiednika elementu
# zerowego z oryginalnych danych.
data = data['Clock_bias'].diff().fillna(0).to_numpy()[1:]

# Normalizujemy dane przesówając je o stałą tak żeby średnia wynosiła
# zero oraz mnożąc przez inną stałą tak żeby największa wartość bezwzględna
# w ciągu wynosiła 1.
mean = data.mean()
data -= mean
scale = max(data.max(), abs(data.min()))
data /= scale

# Żeby zastosować dostrajanie sieci z scikit musimy wygenerować pary
# wejście-wyjście dla ciągu
X = []
Y = []
net_input_len = int(input('Długość wejścia sieci >'))
inputs_count = (len(data)-1) // net_input_len
for i in range(inputs_count):
    shift = i*net_input_len
    X.append(data[shift:shift+net_input_len])
    Y.append(data[shift+1:shift+net_input_len+1])
X = np.asarray(X)
Y = np.asarray(Y)

# Dzielimy teraz te pary na część treningową i testową
tr_count = float(input('Jaka część danych zostanie użyta do uczenia (0;1) >'))
tr_count = floor(X.shape[0]*tr_count)
X_tr = X[:tr_count,:]
Y_tr = Y[:tr_count,:]
X_tst = X[tr_count:,:]
Y_tst = Y[tr_count:,:]
# Reshape wymagany dla warstwy LSTM w kerasie
X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])

print('Wymiary macierzy wejść uczących {}'.format(X_tr.shape))
print('Wymiary macierzy wyjść uczących {}'.format(Y_tr.shape))
print('Wymiary macierzy wejść testowych {}'.format(X_tst.shape))
print('Wymiary macierzy wyjść testowych {}'.format(Y_tst.shape))


# Teraz deklarujemy fukcję generującą sieci neuronowe której scikit
# użyje do tworzenia i testowania różnych sieci

def create_network(optimizer='adam', batch_size=10):
    model = tf.keras.Sequential()
    # Pierwsza warsrwa to LSTM, jej rozmiar musi odpowiadać ilości elementów w sekwencji
    model.add(tf.keras.layers.LSTM(net_input_len,
                                   return_sequences=True,
                                   input_shape=(1, net_input_len))) # FIXME: W miejscu 1 powinien być batch_size
    # Warstwą ukrytą jest warstwa w pełni połączona, może być dowolnego rozmiaru
    model.add(tf.keras.layers.Dense(10))
    # Na sam koniec dajemy drugą warstwę LSTM
    model.add(tf.keras.layers.LSTM(net_input_len, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


# Opakowujemy model kerasa w wrapper dla scikit learn
neural_network = KerasClassifier(build_fn=create_network, verbose=1)

# Definiujemy parametry sieci dla których będziemy testowali
epochs = [1, 5, 10]
batches = [2, 10, 50]
optimizers = ['rmsprop', 'adam']
params = {'epochs':epochs, 'batch_size':batches, 'optimizer':optimizers}

# Tworzymy przeszukiwarkę
grid_search = GridSearchCV(estimator=neural_network, param_grid=params)

search_result = grid_search.fit(X_tr, Y_tr)
print('Najlepsze parametry to : {}'.format(search_result.best_params_))
