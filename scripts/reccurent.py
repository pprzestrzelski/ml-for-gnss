import pandas as pd
import numpy as np
import tensorflow as tf
import sys


class BatchGenerator:

    def __init__(self, data, sequence_length, batch_size, skip=1):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.skip = skip
        self.index = 0

    def generate(self):
        X = np.zeros((self.batch_size, self.sequence_length))
        Y = np.zeros((self.batch_size, self.sequence_length))
        while True:
            for i in range(self.batch_size):
                if self.index + self.sequence_length >= len(self.data):
                    self.index = 0
                X[i, :] = self.data[self.index:self.index + self.sequence_length].reshape(self.sequence_length)
                Y[i, :] = self.data[self.index + 1:self.index + 1 + self.sequence_length].reshape(self.sequence_length)
            print('SHAPE = {}'.format(X.shape))
            X = X.reshape(1, 3, 10)
            yield X, Y



    @staticmethod
    def random_data(sample_size, sequence_length, batch_size):
        data = pd.DataFrame(np.random.random_sample([sample_size, 1]), columns=['Clock_bias'])
        return BatchGenerator(data.values, sequence_length, batch_size)

    @staticmethod
    def load_csv(filename):
        data = pd.read_csv(filename)
        return BatchGenerator(data.values)

def main():
    input_csv = None
    sequence_size = 10
    batch_size = 3
    hidden_size = 10
    gen = BatchGenerator.random_data(100,sequence_size,batch_size)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(sequence_size, return_sequences=True, input_shape=(batch_size,sequence_size)))
    model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
    #model.add(tf.keras.layers.Activation('LeakyReLU'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.fit_generator(gen.generate(), steps_per_epoch=3)

if __name__ == '__main__':
    main()