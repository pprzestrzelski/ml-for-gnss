import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
import keras
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope


# Inicjalizujemy zmienne globalne do późniejszego użytku
# TO JEST ANTIPATTERN :(
data = None
t0 = None
dt = None
e0 = None
mean = None
scale = None
net_input_len = None

# Funkcja służąca do wizualizacji ciągu wartości
def show_visualisation(data, title):
    plt.plot(data)
    plt.ylabel('Wartość')
    plt.xlabel('Numer odczytu')
    plt.title(title)
    plt.show()

# Funkcja do wczytywania danych z CSV
def prepare_data():
    global data,t0,dt,e0,mean,scale
    # Podajemy źródło danych jako pierwszy argument
    data = pd.read_csv(sys.argv[1], sep=';')

    vis = True if input('Wizualizować proces normalizacji (t/n) >')=='t' else False

    t0 = data['Epoch'][0] # Wartość pierwszej epoki
    dt = data['Epoch'][1] - data['Epoch'][0] # Różnice pomiędzy epokami
    e0 = data['Clock_bias'][0] # Bias w zerowym kroku

    print('Pierwsza epoka : {}'.format(t0))
    print('Różnica czasu pomiędzy epokami : {}'.format(dt))
    print('Błąd zegara w pierwszym pomiarze : {}'.format(e0))
    if vis : show_visualisation(data['Clock_bias'].to_numpy(), 'Oryginalne dane')

    # Bierzemy tylko bias i przekształcamy go na różnice pomiędzy kolejnymi
    # wartościami. Oczywiście taki ciąg nie zawiera odpowiednika elementu
    # zerowego z oryginalnych danych.
    data = data['Clock_bias'].diff().fillna(0).to_numpy()[1:]
    if vis : show_visualisation(data, 'Różnice pomiędzy błędami')


    # Normalizujemy dane przesówając je o stałą tak żeby średnia wynosiła
    # zero oraz mnożąc przez inną stałą tak żeby największa wartość bezwzględna
    # w ciągu wynosiła 1.
    mean = data.mean()
    data -= mean
    print('Średnie odchylenie : {}'.format(mean))
    if vis : show_visualisation(data, 'Różnice przesunięte o stałe odchylenie')

    scale = max(data.max(), abs(data.min()))
    data /= scale
    print('Skala normalizacji : {}'.format(scale))
    if vis : show_visualisation(data, 'Różnice po normalizacji')

    return data 


# Funkcja genereuje pary wejście-wyjście dla sieci neuronowej na podstawie
# ciągu. Wyjscie jest zawsze przesunięte względem wejścia o jeden element
def time_series_to_input_and_output(ts, input_len):
    inputs_count = len(ts)-net_input_len
    X=[]
    Y=[]
    for i in range(inputs_count):
        X.append(data[i:i+net_input_len])
        Y.append(data[i+1:i+net_input_len+1])
    return np.asarray(X), np.asarray(Y)


# Funkcja tworząca sieci neuronowe, wymagana do współpracy
# z pakietem scikit learn
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


# Szuka najlepszych parametrów dla sieci neuronowej
def find_best_parameters():
    global net_input_len

    # Żeby zastosować dostrajanie sieci z scikit musimy wygenerować pary
    # wejście-wyjście dla ciągu
    X, Y = time_series_to_input_and_output(data, net_input_len)
    
    # Reshape wymagany dla warstwy LSTM w kerasie
    X = X.reshape(X.shape[0], 1, X.shape[1])

    print('Wymiary macierzy wejść {}'.format(X.shape))
    print('Wymiary macierzy wyjść {}'.format(Y.shape))

    # Opakowujemy model kerasa w wrapper dla scikit learn
    neural_network = KerasClassifier(build_fn=create_network, verbose=0)

    # Definiujemy parametry sieci dla których będziemy testowali
    epochs = [1, 5, 10]
    batches = [2, 10, 50]
    optimizers = ['rmsprop', 'adam']
    params = {'epochs':epochs, 'batch_size':batches, 'optimizer':optimizers}
    
    # Tworzymy przeszukiwarkę
    grid_search = GridSearchCV(estimator=neural_network, param_grid=params)
    
    search_result = grid_search.fit(X, Y)
    print('Najlepsze parametry to : {}'.format(search_result.best_params_))

def teach_and_evaluate():
    global net_input_len

    X, Y = time_series_to_input_and_output(data, net_input_len)

    # Dzielimy teraz te pary na część treningową i testową
    tr_count = float(input('Jaka część danych zostanie użyta do uczenia (0;1) >'))
    tr_count = floor(X.shape[0]*tr_count)
    X_tr = X[:tr_count,:]
    Y_tr = Y[:tr_count,:]
    X_tst = X[tr_count:,:]
    Y_tst = Y[tr_count:,:]
    # Reshape wymagany dla warstwy LSTM w kerasie
    X_tr = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
    X_tst = X_tst.reshape(X_tst.shape[0], 1, X_tst.shape[1])

    print('Wymiary macierzy wejść uczących {}'.format(X_tr.shape))
    print('Wymiary macierzy wyjść uczących {}'.format(Y_tr.shape))
    print('Wymiary macierzy wejść testowych {}'.format(X_tst.shape))
    print('Wymiary macierzy wyjść testowych {}'.format(Y_tst.shape))

    epochs = int(input('Ilość epok > '))
    batch_size = int(input('Rozmiar wsadu (batch) > '))
    optimizer = input('Optymalizator > ')
    nn = create_network(optimizer, batch_size)
    save_best = ModelCheckpoint('best.hdf5', save_best_only=True, monitor='val_loss', mode='min',
                                verbose=1)
    nn.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, validation_data=(X_tst, Y_tst),
           callbacks=[save_best])
    
# Za pomocą tej funkcji będziemy przewidywać ciągi
def predict():
    prediction_depth = int(input('Ile wartości przewidzieć >'))
    inputs_count = len(data)-net_input_len
    X=[]
    # To trzeba tak zrobić z powodu problemów z serializacją glorotuniform
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        nn = keras.models.load_model(sys.argv[2])
        for i in range(inputs_count):
            x = data[i:i+net_input_len]
            x = x.reshape(x.shape[0], 1, 1)
            y = nn.fit()
        X.append(y)
        for _ in prediction_depth:
            x = X[-1]
            x = x.reshape(x.shape[0], 1, 1)
            X.append(nn.fit(x))
            print(X[-1])
    
    
# Opcje do wyboru z menu
option_menu = [find_best_parameters, teach_and_evaluate, predict]
    
# Główna funkcja programu
def main():
    global net_input_len
    net_input_len = int(input('Rozmiar wejścia sieci > '))
    prepare_data()
    
    print('Co chcesz zrobić')
    print('1 - Wyszukać najlepsze parametry sieci')
    print('2 - Nauczyć sieć neuronową')
    print('3 - Przewidzieć wartości błędów')
    print('0 - Zakończyć działanie programu')
    choice = int(input('>'))-1
    if choice == -1 : sys.exit()
    option_menu[choice]()


if __name__ == '__main__':
    main()



