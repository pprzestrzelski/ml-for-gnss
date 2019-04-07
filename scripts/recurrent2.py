import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

#===================================================================================================
#                            Wczytywanie i wizualizacja danych z CSV
#===================================================================================================

# Obiekty tej klasy zawierają ciąg znormalizowanych różnic pomiędzy odchyleniami
# zegara oraz informacje umożliwiające odtworzenie pliku csv.
class ErrorData:

    def __init__(self, t0, dt, e0, scale, raw_data):
        self.t0 = t0 # Wartość czasu (epoch) dla pierwszego elementu ciągu
        self.dt = dt # Różnica czasu pomiędzy odczytami
        self.e0 = e0 # Watość błędu dla pierwszego elementu (różnica błędu będzie zawsze zerowa)
        self.scale = scale # Prawdziwa różnica to wartość w sekwencji pomnożona przez skalę
        self.raw_data = raw_data # Ciąg znormalizowanych różnic pomiędzy błędami odczytu

    @staticmethod
    def load_csv(filename, scale=None):
        data = pd.read_csv(filename)
        t0 = data['Epoch'][0]
        dt = data['Epoch'][1] - data['Epoch'][0]
        e0 = data['Clock_bias'][0]
        # Zmieniamy wartości błędów na różnice pomiędzy tymi wartościami, normalnie
        # pierwszy element ciągu będzie NaN więc musimy zmienić go na 0
        raw_data = data['Clock_bias'].diff().fillna(0)
        # Normalizacja, robimy to ręcznie zamiast scalerem dla większej kontroli
        if scale is None: scale = data['Clock_bias'].max() - data['Clock_bias'].min()
        raw_data = raw_data.to_numpy() / scale
        return ErrorData(t0, dt, e0, scale, raw_data)

# Wizualizacja różnic pomiędzy błędami za pomoca pyplot
def visualise_error_data(ed):
    plt.plot(ed.raw_data)
    plt.ylabel('Normalized error difference')
    plt.xlabel('Readout number')
    plt.show()



#===================================================================================================
#                                   Generatory danych dla Kerasa
#===================================================================================================

# Klasa generująca wejścia dla kerasa podczas uczenia
class TrainingDataBatch:

    def __init__(self, ed, seq_len, step, batch_size):
        self.ed = ed # ErrorData
        self.seq_len = seq_len # Długość sekwencji podawanej na wejście sieci
        self.step = step # O ile elementów przesówamy sekwencje pomiędzy cyklami sieci
        self.batch_size = batch_size # Ile cylki uczący zostanie wykożystanych w jednym pokoleniu
        self.idx = 1 # Obecna pozycja w ciągu, pomijamy element zerowy poniewaz będzie zakłócać uczenie

    # Ta funkcja jest używana przez kerasa i zawsze musi zwracać parę tablic numpy,
    # wejścia i wyjścia sieci
    def generate(self):
        data = self.ed.raw_data # Dla poprawy czytelności, inaczej linijki będą bardzo długie
        while True:
            X = np.zeros((self.batch_size, self.seq_len))
            Y = np.zeros((self.batch_size, self.seq_len))
            # Generujemy kolejno pary wejście-wyjscie
            for i in range(self.batch_size):
                # Jeżeli dotarliśmy do końca powracamy do pierwszego (! nie zerowego bo ten ignorujemy!)
                # elementu
                if self.idx + self.step + self.seq_len > len(data): self.idx = 1
                # Wyjście jest przesunięte w ciągu względem wejścia o wartość step
                X[i,:] = data[self.idx:self.idx+self.seq_len].reshape(self.seq_len)
                Y[i,:] = data[self.idx+self.step:self.idx+self.step+self.seq_len].reshape(self.seq_len)
            # Musimy jeszcze przekształcić nasze macierze na trójwymiarowe bo takich oczekuje keras
            X = X.reshape(1, self.batch_size, self.seq_len)
            Y = Y.reshape(1, self.batch_size, self.seq_len)
            yield X, Y
        

#===================================================================================================
#                                   Właściwa sieć neuronowa
#===================================================================================================

# Funkcja wczytująca proste pliki konfiguracyjne w styly ".properties"
def load_config(filename):
    cfg = {}
    try:
        with open(filename, 'r') as cfg_file:
            for line in cfg_file:
                if line[0] == '#': continue
                key, value = line.split('=')
                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass
                cfg[key] = value
    except Exception as e:
        print('Error occured when loading configuration file.')
        print(e)
    return cfg


# Klasa zawiera w sobie wszystkie funkcjonalności sieci neuronowej
class NeuralNetwork:

    def __init__(self, model):
        self.model = model


    @staticmethod
    def build_lstm(cfg_file, weight_file=None):
        cfg = load_config(cfg_file)
        model = tf.keras.Sequential()
        # Pierwsza warsrwa to LSTM, jej rozmiar musi odpowiadać ilości elementów w sekwencji
        model.add(tf.keras.layers.LSTM(cfg['sequence_size'],
                                       return_sequences=True,
                                       input_shape=(cfg['batch_size'], cfg['sequence_size'])))
        # Warstwą ukrytą jest warstwa w pełni połączona, może być dowolnego rozmiaru
        model.add(tf.keras.layers.Dense(cfg['hidden_size']))
        # Na sam koniec dajemy drugą warstwę LSTM
        model.add(tf.keras.layers.LSTM(cfg['sequence_size'], return_sequences=True))
        if weight_file is not None:
            model.load_weights(weight_file)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        return NeuralNetwork(model)

    # Uczenie sieci
    def fit(self, data_batch, cfg_file, weight_folder='checkpoints'):
        cfg = load_config(cfg_file)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=weight_folder + '/model-{epoch:02d}.hdf5', verbose=1)
        self.model.fit_generator(data_batch.generate(),
                                 steps_per_epoch=cfg['steps_per_epoch'],
                                 epochs=cfg['epochs'],
                                 callbacks = [checkpointer])


# Główna funkcja skryptu
def main():
    ed = ErrorData.load_csv(sys.argv[1])
    visualise_error_data(ed)
    tb = TrainingDataBatch(ed,10,1,3)
    net = NeuralNetwork.build_lstm('configs/demo.txt')
    net.fit(tb,'configs/demo.txt')
    
if __name__ == '__main__':
    main()
