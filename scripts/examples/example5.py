from core.gnss.gnss_clock_data import GnssClockData
import tensorflow as tf
import numpy as np

INPUT_SIZE = 10
HIDDEN_SIZE = 512

def buid_input_output_pairs(sequence, input_size):
    input_chunks = [sequence[x:x + input_size] for x in range(0, len(sequence), input_size)]
    #output_chunks = [sequence[(x+1):(x+1+input_size)] for x in range(0, len(sequence), input_size)]
    output_chunks = []
    for i in range(0, len(sequence), input_size):
        output_chunks.append(sequence[i:i+input_size])
    print(input_chunks[0])
    print(output_chunks[0])
    input_chunks = np.asarray(input_chunks)
    print(input_chunks.shape)
    input_chunks = np.array(input_chunks)[:, :, np.newaxis]
    output_chunks = np.asarray(output_chunks)
    print(output_chunks.shape)
    output_chunks = np.array(output_chunks)[:, :, np.newaxis]

    return input_chunks, output_chunks


def teach_network(inputs, outputs):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        tf.keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.relu),
        tf.keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(inputs, outputs, epochs=5)


def main():
    clock_data = GnssClockData(dir_name="clock_data")
    sat_number = 'G05'
    data = clock_data.get_satellite_data(sat_number)
    data = [x[1].bias for x in data]
    train_in, train_out = buid_input_output_pairs(data, INPUT_SIZE)
    teach_network(train_in, train_out)
    print('Loaded data type : {}'.format(type(data)))
    print('Loaded data length : {}'.format(len(data)))
