# In here I will store some experiments

from core.gnss.gnss_clock_data import GnssClockData


def main():
    clock_data = GnssClockData(dir_name="clock_data")

    # Choose satellite to investigate
    # (interesting data: G05, G23, G24)
    sat_number = 'G05'

    # Split data (epoch_data, clock_data) manually
    data = clock_data.get_satellite_data(sat_number)
    print(data)


if __name__ == '__main__':
    main()