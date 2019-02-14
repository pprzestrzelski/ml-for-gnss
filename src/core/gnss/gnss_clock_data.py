import os
from sortedcontainers import SortedDict
from src.core.gnss.rinex_clock_file import RinexClockFile


class GnssClockData:
    def __init__(self, dir_name="clock_data", file_standard="RINEX", add_all=True):
        self.dir_name = dir_name
        self.file_standard = file_standard
        self.files = []
        self.data = SortedDict()
        if add_all:
            self.__add_all_data_from_dir()

    def __add_all_data_from_dir(self):
        print("=== Reading clock data file in {} format...".format(self.file_standard))
        for file in os.listdir(self.dir_name):
            if self.file_standard == "RINEX" and (file.endswith(".clk") or file.endswith(".clk_30s")):
                self.files.append(file)
                self.add_rinex_file(self.dir_name + '/' + file)
            elif self.file_standard == "SP3" and file.endswith(".sp3"):
                # TODO: implement in the future
                print("SP3 data file", file, "omitted! (NOT IMPLEMENTED YET!!!)")
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

    def read_clock_data(self):
        self.__add_all_data_from_dir()

    def files_in_memory(self):
        return len(self.data)

    def get_satellite_data(self, sat):
        data = []
        for _, value in self.data.items():
            data.extend(value.get_data(sat))

        return data

    # def split_data(self, data):
    #     epochs = []
    #     clock_data = []
    #     for i in range(len(data)):
