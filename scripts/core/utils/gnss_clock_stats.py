from core.gnss.gnss_clock_data import GnssClockData
from core.utils.gnss_global_consts import GPS_SATS


class TrivialGnssClockStats:
    def __init__(self):
        self.clock_data = None
        self.data_interval = 0
        self.length = 24
        self.sp3_data_type = "Observed"

    def add_data(self, data, interval_in_sec, length_in_hours=24):
        if isinstance(data, GnssClockData):
            self.clock_data = data
            self.data_interval = interval_in_sec
            self.length = length_in_hours
        else:
            print("WARNING: {} is not a GnssClockData file".format(data))

    def set_sp3_data_type(self, data_type):
        if data_type not in ["Observed", "Predicted", "Both"]:
            print("WARNING: invalid data type \"{}\"".format(data_type))
        else:
            self.sp3_data_type = data_type

    def print_missing(self):
        epochs_total = self.clock_data.files_in_memory() * 24 * 60 * 60 / self.data_interval
        print("=== Missing GNSS clock data (%)")
        for sat in GPS_SATS:
            if self.clock_data.file_standard == "RINEX":
                data = self.clock_data.get_satellite_data(sat)
            elif self.clock_data.file_standard == "SP3":
                data = self.clock_data.get_satellite_data(sat, data_type=self.sp3_data_type)
            else:
                data = []
            epochs_counted = len(data)
            diff = epochs_total - epochs_counted
            if diff > 0:
                print("   -> {0}: {1:.2f}% ({2} epochs)".format(sat, float(diff/epochs_total) * 100.0, int(diff)))
