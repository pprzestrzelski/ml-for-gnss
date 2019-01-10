from gnss_clock_data import GnssClockData
import matplotlib.pyplot as plt

SCALE = 10.0 ** 9   # ns    1ns => ~30 cm

if __name__ == '__main__':
    clock_data = GnssClockData()

    # Ciekawe dane: G05, G23, G24
    sat_number = 'G01'
    data = clock_data.get_satellite_data(sat_number)

    epochs = []
    clock_biases = []
    for i in range(len(data)):
        epochs.append(data[i][0])
        clock_biases.append(float(data[i][1].get_bias()) * SCALE)

    plt.plot([i for i in range(len(clock_biases))], clock_biases, 'o')
    plt.title('Poprawki chodu zegara satelity {}'.format(sat_number))
    plt.ylabel('[ns]')

    dy = (max(clock_biases) - min(clock_biases)) * 0.1
    plt.ylim(min(clock_biases) - dy, max(clock_biases) + dy)

    plt.show()
