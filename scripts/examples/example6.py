from core.gnss.gnss_clock_data import GnssClockData
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

INPUT_SIZE = 10
HIDDEN_SIZE = 512


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def prepare_data(sat_number, input_length=10, buffer_size=10000, batch_size=20):
    clock_data = GnssClockData(dir_name="clock_data")
    sequence = clock_data.get_satellite_data(sat_number)
    sequence = [float(x[1].bias) for x in sequence]

    examples_per_epoch = len(sequence) #  Important setting
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    sequences = dataset.batch(input_length + 1, drop_remainder=True)
    #for i, seq in enumerate(sequences.take(5)):
    #    print('Sequence {} : {}'.format(i, seq.numpy()))

    sequences = dataset.batch(input_length + 1, drop_remainder=True)
    #for i, seq in enumerate(sequences.take(5)):
    #    print('Sequence {} : {}'.format(i, seq.numpy()))

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print('{}'.format(dataset.output_shapes))
    return dataset, examples_per_epoch // batch_size


def prepare_network(input_length=10, network_type='LSTM', dense_size=10):
    input_layer = None
    if network_type == 'ff':
        input_layer = tf.keras.layers.Dense(input_length, activation=tf.nn.relu)
    elif network_type == 'lstm':
        input_layer = tf.keras.layers.LSTM(input_length, input_shape=(10, 1))

    if input_layer is None:
        raise Exception('Unknown network type : {}'.format(network_type))

    dense = tf.keras.layers.Dense(dense_size)
    network = tf.keras.Sequential([input_layer, dense])
    network.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy']
                  )

    return network


def main():
    dataset, steps_per_epoch = prepare_data('G05')
    network_type = input('Enter network type : ')
    network = prepare_network(network_type=network_type)
    print('DATASET TYPE = {}'.format(type(dataset).__name__))
    if input('Train network (y/n):') == 'y':
        network.fit(dataset.repeat(), epochs=5,
            steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    main()
