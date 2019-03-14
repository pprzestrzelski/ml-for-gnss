from core.gnss.gnss_clock_data import GnssClockData
from core.utils.gnss_clock_stats import TrivialGnssClockStats


def main():
    clock_data = GnssClockData(dir_name="clock_data")
    stats = TrivialGnssClockStats()
    stats.add_data(clock_data, 30)
    stats.print_missing()


if __name__ == "__main__":
    main()
