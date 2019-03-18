import pandas as pd
import numpy as np
import tensorflow as tf
import sys

class Config:

    def __init__(self, sequence_size, batch_size, hidden_size, sequence_shift,
                 loss_function, steps_per_epoch, epochs):
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.sequence_shift = sequence_shift
        self.loss_function = loss_function
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs


    @staticmethod
    def demo_config():
        return Config(10, 3, 10, 1, 'mean_squared_error', 100, 10)


class BatchGenerator:

    def __init__(self, data, sequence_length, batch_size, skip=1):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.skip = skip
        self.index = 0

    def generate(self):
        while True:
            X = np.zeros((self.batch_size, self.sequence_length))
            Y = np.zeros((self.batch_size, self.sequence_length))
            for i in range(self.batch_size):
                if self.index + self.sequence_length >= len(self.data):
                    self.index = 0
                X[i, :] = self.data[self.index:self.index + self.sequence_length].reshape(self.sequence_length)
                Y[i, :] = self.data[self.index + 1:self.index + 1 + self.sequence_length].reshape(self.sequence_length)
            X = X.reshape(1, self.batch_size, self.sequence_length)
            Y = Y.reshape(1, self.batch_size, self.sequence_length)
            yield X, Y



    @staticmethod
    def random_data(sample_size, sequence_length, batch_size):
        data = pd.DataFrame(np.random.random_sample([sample_size, 1]), columns=['Clock_bias'])
        return BatchGenerator(data.values, sequence_length, batch_size)

    @staticmethod
    def load_csv(filename):
        data = pd.read_csv(filename)
        return BatchGenerator(data.values)


class NeuralNetwork:

    def __init__(self, model):
        self.model = model

    def fit(self, generator, cfg):
        # FIXME: Make checkpoints work
        #checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkopints' + '/model-{epoch:02d}.hdf5', verbose=1)
        self.model.fit_generator(generator.generate(),
                                 steps_per_epoch=cfg.steps_per_epoch,
                                 epochs=cfg.epochs)

    @staticmethod
    def build_lstm_model(cfg):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(cfg.sequence_size,
                                       return_sequences=True,
                                       input_shape=(cfg.batch_size, cfg.sequence_size)))
        model.add(tf.keras.layers.LSTM(cfg.hidden_size, return_sequences=True))
        model.compile(loss=cfg.loss_function, optimizer='adam', metrics=['mae'])
        return NeuralNetwork(model)

def main():
    input_csv = None
    cfg = Config.demo_config()
    gen = BatchGenerator.random_data(1000, cfg.sequence_size, cfg.batch_size)
    nn = NeuralNetwork.build_lstm_model(cfg)
    nn.fit(gen, cfg)



if __name__ == '__main__':
    main()