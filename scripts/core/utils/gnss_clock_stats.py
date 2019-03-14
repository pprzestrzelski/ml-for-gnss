from core.gnss.gnss_clock_data import GnssClockData
from core.utils.gnss_global_consts import GPS_SATS


class TrivialGnssClockStats:
    def __init__(self):
        self.clock_data = None
        self.data_interval = 0
        self.length = 24

    def add_data(self, data, interval_in_sec, length_in_hours=24):
        if isinstance(data, GnssClockData):
            self.clock_data = data
            self.data_interval = interval_in_sec
            self.length = length_in_hours
        else:
            print("WARNING: {} is not a GnssClockData file".format(data))

    def print_missing(self):
        epochs_total = self.clock_data.files_in_memory() * 24 * 60 * 60 / self.data_interval
        print("=== Missing GNSS clock data (%)")
        for sat in GPS_SATS:
            epochs_counted = len(self.clock_data.get_satellite_data(sat))
            diff = epochs_total - epochs_counted
            if diff > 0:
                print("   -> {0}: {1:.2f}% ({2} epochs)".format(sat, float(diff/epochs_total) * 100.0, int(diff)))
