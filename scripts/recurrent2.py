import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
import tensorflow as tf
import numpy as np
import argparse as ap
import sys
import logging 
from contextlib import suppress

# Logger którego będziemy używać dla wszystkich wiadomości
log = logging.getLogger('global_logger')

# ===================================================================================================
#                            Wczytywanie i wizualizacja danych z CSV
# ===================================================================================================

# Obiekty tej klasy zawierają ciąg znormalizowanych różnic pomiędzy odchyleniami
# zegara oraz informacje umożliwiające odtworzenie pliku csv.
class ErrorData:

    def __init__(self, t0, dt, e0, mean, scale, raw_data):
        self.t0 = t0  # Wartość czasu (epoch) dla pierwszego elementu ciągu
        self.dt = dt  # Różnica czasu pomiędzy odczytami
        self.e0 = e0  # Watość błędu dla pierwszego elementu (różnica błędu będzie zawsze zerowa)
        self.scale = scale  # Obiek do normalizacji danych
        self.raw_data = raw_data  # Ciąg znormalizowanych różnic pomiędzy błędami odczytu

    @staticmethod
    def load_csv(filename, mean=None, scale=None):
        data = pd.read_csv(filename, sep=';')
        t0 = data['Epoch'][0]
        dt = data['Epoch'][1] - data['Epoch'][0]
        e0 = data['Clock_bias'][0]
        # Zmieniamy wartości błędów na różnice pomiędzy tymi wartościami, normalnie
        # pierwszy element ciągu będzie NaN więc musimy zmienić go na 0
        raw_data = data['Clock_bias'].diff().fillna(0).to_numpy()[1:]
        # Normalizacja za pomocą StandardScalera, ręcznie były błędy
        if mean is None: mean = raw_data.mean()
        raw_data -= mean
        if scale is None: scale = max(raw_data.max(), abs(raw_data.min()))
        raw_data /= scale
        return ErrorData(t0, dt, e0, mean, scale, raw_data)

    def save_csv(self, filename):
        # Skalujemy z powrotem do oryginalnych wartości
        data = self.raw_data * self.scale + self.mean
        data.insert(0, self.e0)  # Ustawiamy wstępny błąd jako pierwszą wartość
        data = np.cumsum(data)  # Każda wartość będzie teraz sumą wszystkich poprzednich
        # Tworzymy tablicę z wartościami zaczynającymi się od zerowej epoki a następnie
        # inkrementowanymi o wartość przyrostu czasu. Tablica będzie miała taki sam
        # rozmiar jak data
        epochs = [t0]
        # TODO : To jest napisane paskudnie, niepythonowo oraz niewydajnie.
        # ale na szybko.
        for _ in range(len(data)-1):
            epochs.append(epochs[-1]+dt)
        epochs = np.array(epochs)
        d_frame = {'Epoch': epochs, 'Clock_bias': data}
        df = pd.DataFrame(d_frame, columns=['Epoch', 'Clock_bias'])
        df.to_csv(filename, index=False, sep=';')  # Zapisz do pliku csv bez numerowania wierszy


# Wizualizacja różnic pomiędzy błędami za pomoca pyplot
def visualise_error_data(ed):
    plt.plot(ed.raw_data)
    plt.ylabel('Normalized error difference')
    plt.xlabel('Readout number')
    plt.show()

def visualise_error_data_raw(raw_data):
    plt.plot(raw_data)
    plt.ylabel('Normalized error difference')
    plt.xlabel('Readout number')
    plt.show()


# ===================================================================================================
#                                   Generatory danych dla Kerasa
# ===================================================================================================

# Klasa generująca wejścia dla kerasa podczas uczenia
class TrainingDataBatch:

    def __init__(self, ed, seq_len, step, batch_size):
        self.ed = ed  # ErrorData
        self.seq_len = seq_len  # Długość sekwencji podawanej na wejście sieci
        self.step = step  # O ile elementów przesówamy sekwencje pomiędzy cyklami sieci
        self.batch_size = batch_size  # Ile cylki uczący zostanie wykożystanych w jednym pokoleniu
        self.idx = 0  # Obecna pozycja w ciągu

    # Ta funkcja jest używana przez kerasa i zawsze musi zwracać parę tablic numpy,
    # wejścia i wyjścia sieci
    def generate(self):
        data = self.ed.raw_data  # Dla poprawy czytelności, inaczej linijki będą bardzo długie
        while True:
            X = np.zeros((self.batch_size, self.seq_len))
            Y = np.zeros((self.batch_size, self.seq_len))
            # Generujemy kolejno pary wejście-wyjscie
            for i in range(self.batch_size):
                # Jeżeli dotarliśmy do końca powracamy do pierwszego (! nie zerowego bo ten ignorujemy!)
                # elementu
                if self.idx + self.step + self.seq_len > len(data):
                    self.idx = 1
                # Wyjście jest przesunięte w ciągu względem wejścia o wartość step
                X[i, :] = data[self.idx:self.idx+self.seq_len].reshape(self.seq_len)
                Y[i, :] = data[self.idx+self.step:self.idx+self.step+self.seq_len].reshape(self.seq_len)
            # Musimy jeszcze przekształcić nasze macierze na trójwymiarowe bo takich oczekuje keras
            X = X.reshape(1, self.batch_size, self.seq_len)
            Y = Y.reshape(1, self.batch_size, self.seq_len)
            yield X, Y
        

# Klasa generująca wejścia dla kerasa podczas predykcji, będzie generować wejścia do chwili w
# której głębokość predykcji (ilość wartości wygenerowanych przez sieć) nie osiągnie zadanego
# limitu.
# TODO: To napewno da się ładniej zrobić
class PredictionDataBatch:

    def __init__(self, base, seq_len, step, prediction_depth):
        self.base = base  # Ciąg odchyleń stanowiący podstawę dla predykcji
        self.step = step  # O ile elementów przesówamy sekwencje pomiędzy cyklami sieci
        self.seq_len = seq_len  # Długość sekwencji podawanej na wejście sieci
        self.prediction_depth = prediction_depth  # Ilość wartości która ma zostać przewidziana
        self.predicted_count = 0  # Ilość już przewidzianych elementów
        self.idx = 1  # Pozycja pierwszego elementu wejścia sieci w ciągu
        self.predicted = []  # Elementy przewidziane (nie używać len(predicted) !!!!)

    # Dzięki tej funkcji możemy łatwo iterować bo obiekcie za pomocą funkcji for
    def __iter__(self):
        return self

    # Ta funkcja faktycznie ogarnia iterowanie
    def __next__(self):
        #log.debug('Krok iteracji predyktora')
        if self.predicted_count >= self.prediction_depth:
            log.debug('Osiągnięto zamierzoną głębokość predykcji')
            raise StopIteration
        dat = self.base.raw_data  # Dla lepszej czytelności
        X = None
        # Sprawdzamy czy wykożystujemy jeszcze bazę
        if self.idx < len(dat):
            # Jeżeli wykożystujemy tylko bazę to nie ma tu zbyt dużej filozofi
            if self.idx+self.seq_len < len(dat):
                X = dat[self.idx:self.idx+self.seq_len].reshape(1, self.seq_len)
                self.idx += self.step
            else:
                from_base = dat[self.idx:]  # Pobieramy wszystko co możemy z bazy
                # Potem dobieramy do tego elementy z przewidzianych
                from_prediction = np.asarray(self.predicted, dtype=np.float64)
                # TODO : Przyjmujemy śmiałe założenie że te dwie listy mają łącznie
                # długość odpowiadająca oczekiwanej długości wejścia.
                # To możeokazać się nieprawdziwe.
                log.debug('from_base={} from_prediction={}'.format(from_base.shape,
                                                                   from_prediction.shape))
                X = np.hstack([from_base, from_prediction])
                self.idx += self.step
                self.predicted_count += self.step
        else:
            # Ponieważ nie używamy już bazy trzeba odjąć jej rozmmiar od indeksu
            local_idx = self.idx - len(dat)
            X = self.predicted[local_idx:local_idx+self.seq_len]
            self.idx += self.step
            self.predicted_count += self.step
        return np.asarray(X).reshape(1, 1, self.seq_len)

    # Dodajemy wartości przewidziane z wyjścia sieci neuronowej
    def update(self, predictions):
        update_size = self.predicted_count > len(self.predicted)
        if update_size > 0:
            # Predictions jest maicerzą 3-wymiarową (keras tak ma) więc trzeba wydobyć
            # interesujące nas dane TODO: Jak będziemy pracować na większych wsadach
            # trzeba będzie tu poprawić.
            self.predicted.append(predictions[0][0][-update_size])

    def build_error_data(self, concatenate_base=True):
        dat = None
        t0 = self.base.t0
        e0 = self.base.e0
        if concatenate_base:
            dat = np.hstack([self.base.raw_data, np.asarray(self.predicted, dtype=np.float64)])
        else:
            dat = np.asarray(self.predicted, dtype=np.float64)
            # Jeżeli nie dołączamy danych bazowych należy uaktualnić wartości początkowe
            # tak żeby pokrywały się z stanem na końcu bazy
            t0 += self.base.dt * len(self.base.raw_data)
            e0 += (self.base.raw_data*self.base.scale).sum()
        return ErrorData(t0, self.base.dt, e0, self.base.scaler, dat)


# ===================================================================================================
#                                   Właściwa sieć neuronowa
# ===================================================================================================

# Klasa zawiera w sobie wszystkie funkcjonalności sieci neuronowej
class NeuralNetwork:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def build_lstm(sequence_size, batch_size, hidden_size, weight_file=None):
        model = tf.keras.Sequential()
        # Pierwsza warsrwa to LSTM, jej rozmiar musi odpowiadać ilości elementów w sekwencji
        model.add(tf.keras.layers.LSTM(sequence_size,
                                       return_sequences=True,
                                       input_shape=(batch_size, sequence_size)))
        # Warstwą ukrytą jest warstwa w pełni połączona, może być dowolnego rozmiaru
        model.add(tf.keras.layers.Dense(hidden_size))
        # Na sam koniec dajemy drugą warstwę LSTM
        model.add(tf.keras.layers.LSTM(sequence_size, return_sequences=True))
        if weight_file is not None:
            model.load_weights(weight_file)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        return NeuralNetwork(model)

    # Uczenie sieci
    def fit(self, data_batch, steps_per_epoch, epochs, weight_folder='checkpoints'):
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=weight_folder + '/weights.hdf5',
                                                          save_best_only=True,
                                                          verbose=1)
        self.model.fit_generator(data_batch.generate(),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 callbacks=[checkpointer])

    # Opakowanie funkcji przewidującej
    def predict(self, X):
        return self.model.predict(X)


# ===================================================================================================
#                                    Wczytywanie konfiguracji
# ===================================================================================================

# Staramy się przekonwetować napis do jak najbardziej restrykcyjnego typu czyli po kolei
# int, float, string
def parse_config_value(string):
    # Ignorujemy wyjątki typu ValueError
    with suppress(ValueError):
        return int(string)
        return float(value)
        return string


# Funkcja wczytująca proste pliki konfiguracyjne w styly ".properties"
def load_config(filename):
    cfg = {}
    try:
        with open(filename, 'r') as cfg_file:
            for line in cfg_file:
                if line[0] == '#':
                    continue
                key, value = line.split('=')
                value = parse_config_value(value)
                cfg[key] = value
    except Exception as e:
        print('Error occured when loading configuration file.')
        print(e)
    return cfg

# Wypisuje konfigurację
def print_config(cfg):
    log.info('Configuration : ')
    for k, v in cfg.items():
        log.info('{}:{} ({})'.format(k, v, type(v).__name__))
    

# Funkcja przygotowuje parser dla linii poleceń
def prepare_argparser():
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--input", help="Path to clock csv data",
                        type=str)
    parser.add_argument('-o', "--output", help="Name of file to which output will be saved in prediction mode",
                        type=str)
    parser.add_argument('-c', "--config", help="Path to configuration file",
                        type=str)
    parser.add_argument('-w', "--weights", help="Path to weights file",
                        type=str, default=None)
    parser.add_argument('-t', "--train", help="Train network on given data",
                        action='store_true')
    parser.add_argument('-e', "--evaluate", help="Evaluate network on given data",
                        action='store_true')
    parser.add_argument('-p', "--predict", help="Predict future errors from given data",
                        action='store_true')
    parser.add_argument('-d', "--depth", help="Number of readouts that will be predicted",
                        type=int)
    return parser
    

# Główna funkcja skryptu
def main():
    args = prepare_argparser().parse_args()
    logging.basicConfig(level=logging.DEBUG)
    ed = ErrorData.load_csv(args.input)
    visualise_error_data(ed)
    cfg = load_config(args.config)
    print_config(cfg)
    net = NeuralNetwork.build_lstm(cfg['sequence_size'], cfg['batch_size'], cfg['hidden_size'], args.weights)
    if args.train:
        tb = TrainingDataBatch(ed, cfg['sequence_size'], cfg['step'], cfg['batch_size'])
        net.fit(tb, cfg['steps_per_epoch'], cfg['epochs'])
    elif args.predict:
        pb = PredictionDataBatch(ed, cfg['sequence_size'], cfg['step'], args.depth)
        for X in pb:
            pb.update(net.predict(X))
        pb.build_error_data().save_csv(args.output)


if __name__ == '__main__':
    main()
