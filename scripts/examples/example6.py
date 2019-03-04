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
    for i, seq in enumerate(sequences.take(5)):
        print('Sequence {} : {}'.format(i, seq.numpy()))

    sequences = dataset.batch(input_length + 1, drop_remainder=True)
    for i, seq in enumerate(sequences.take(5)):
        print('Sequence {} : {}'.format(i, seq.numpy()))

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset, examples_per_epoch // batch_size


def prepare_network(input_length=10, dense_size=30):
    rnn = None
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')

    recurrent = rnn(input_length,
                    return_sequences=True,
                    recurrent_initializer='glorot_uniform',
                    stateful=True)
    dense = tf.keras.layers.Dense(dense_size)
    network = tf.keras.Sequential([recurrent, dense])

    return network


def main():
    dataset, steps_per_epoch = prepare_data('G05')
    network = prepare_network()
    network.fit(dataset.repeat(), epochs=5,
        steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    main()
