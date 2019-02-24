import os
from sortedcontainers import SortedDict
from core.gnss.rinex_clock_file import RinexClockFile
from core.gnss.sp3_file import Sp3File


class GnssClockData:
    """
    GnssClockData is a container for a satellite clock data from RINEX or SP3 file formats.
    In this moment file standard mix is not allowed.
    """
    def __init__(self, dir_name="clock_data", file_standard="RINEX", add_all=True):
        """
        :param dir_name: path to directory with GNSS clock data files
        :param file_standard: RINEX or SP3 file to read
        :param add_all: read automatically when object is created
        """
        self.dir_name = dir_name
        self.file_standard = file_standard
        self.files = []
        self.data = SortedDict()
        if add_all:
            self.__add_all_data_from_dir()

    def __add_all_data_from_dir(self):
        print("=== Reading clock data file in {} format...".format(self.file_standard))
        for file in os.listdir(self.dir_name):
            if file in self.files:
                print("{} is in memory. It will be skipped.".format(file))
                continue

            if self.file_standard == "RINEX" and (file.endswith(".clk") or file.endswith(".clk_30s")):
                self.files.append(file)
                self.add_rinex_file(self.dir_name + '/' + file)
            elif self.file_standard == "SP3" and file.endswith(".sp3"):
                self.files.append(file)
                self.add_sp3_file(self.dir_name + '/' + file)
            elif os.path.isdir(self.dir_name + "/" + file):
                print("{} is directory... omitting it".format(file))
            else:
                print("Unknown file format (>", file, "<)")
        print("=== all files read")

    def add_rinex_file(self, file_path):
        clock_data = RinexClockFile(file_path)
        if clock_data.first_epoch() in self.data:
            print("WARNING: this file ({}) is in the database. It will not be proceeded.".format(file_path))
        else:
            self.data[clock_data.first_epoch()] = clock_data
            print("Added file {}".format(file_path))

    def add_sp3_file(self, file_path):
        clock_data = Sp3File(file_path)
        if clock_data.first_epoch() in self.data:
            print("WARNING:this file ({}) is in the database. It will not be proceeded.".format(file_path))
        else:
            self.data[clock_data.first_epoch()] = clock_data
            print("Added file {}".format(file_path))

    def read_clock_data(self):
        self.__add_all_data_from_dir()

    def files_in_memory(self):
        return len(self.data)

    def get_satellite_data(self, sat, data_type="Observed"):
        """
        Reads GNSS clock data from all found files.
        :param sat: satellite data to read, e.g. 'G01' as GPS satellite #01
        :param data_type: type of data read: 'Observed', 'Predicted' or 'Both' (concerns only SP3 standard)
        :return: array of pairs of data: [(epoch, clock_bias), (..., ...), ...]
        """
        data = []
        if self.file_standard == "RINEX":
            for _, value in self.data.items():
                data.extend(value.get_data(sat))
        elif self.file_standard == "SP3":
            for _, value in self.data.items():
                data.extend(value.get_data(sat, data_type))

        return data

    # def split_data(self, data):
    #     epochs = []
    #     clock_data = []
    #     for i in range(len(data)):
