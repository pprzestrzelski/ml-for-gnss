from core.gnss.gnss_clock_data import GnssClockData
from core.utils.gnss_global_consts import GPS_SATS
import time
import os


class ClockDataConverter:
    def __init__(self, dir_name="default"):
        self.__data = None
        self.__dir = "conversion_" + time.strftime("%Y%m%d_%H%M%S") if dir_name == "default" else dir_name
        self.dir_raw_csv = ""
        self.dir_modified_csv = ""
        self.__create_dir_structure()
        self.csv_sep = ';'
        self.sp3_data_type = "Observed"

    def __create_dir_structure(self):
        """
        Creates directory data structure in a current working directory (CWD) as follows:
            - top_dir ("conversion_20190310_201920")
                + raw_csv ("raw_csv") - contains raw clock data read
                    from GnssClockData converted (one csv files per sat)
                + modified_csv ("modified_csv") - has to have clock data in exactly
                    the same format as files in "raw_csv"
        ATTENTION: clock biases in csv files are given in ns!!!
        """
        cwd = os.getcwd()
        top_dir = cwd + os.sep + "conversions"
        perm = 0o755
        try:
            if not os.path.exists(top_dir):
                os.mkdir(top_dir, perm)
            else:
                print('Top directory exists ({})'.format(top_dir))
        except OSError as e:
            print("ERROR: could not create top directory! (error: {})".format(e))
            return

        conv_dir = top_dir + os.sep + self.__dir
        try:
            os.mkdir(conv_dir, perm)
        except OSError as e:
            print("ERROR: could not create directory {} for file conversions (error: {})".format(conv_dir, e))
            return

        try:
            sub_dir_raw = conv_dir + os.sep + "raw_csv"
            os.mkdir(sub_dir_raw, perm)
            sub_dir_modified = conv_dir + os.sep + "modified_csv"
            os.mkdir(sub_dir_modified, perm)
            self.dir_raw_csv = sub_dir_raw
            self.dir_modified_csv = sub_dir_modified
        except OSError as e:
            print("WARNING: something went wrong. Could not create sub-dirs (error: {})".format(e))
        print(self.dir_modified_csv)

    def add_data(self, gnss_clock_data):
        if isinstance(gnss_clock_data, GnssClockData):
            self.__data = gnss_clock_data
            print("Clock data added to the converter ({} file(s))".format(gnss_clock_data.files_in_memory()))
        else:
            print("WARNING: given file is not a GnssClockData file!")

    def set_sp3_data_type(self, data_type):
        if data_type not in ["Observed", "Predicted", "Both"]:
            print("WARNING: invalid data type \"{}\"".format(data_type))
        else:
            self.sp3_data_type = data_type

    def data_to_csv(self):
        scale = 10.0 ** 9 if self.__data.file_standard == "RINEX" else 10.0 ** 3
        for sat in GPS_SATS:
            csv_path = self.dir_raw_csv + os.sep + sat + ".csv"
            with open(csv_path, 'w') as csv_file:
                header = "Epoch" + self.csv_sep + "Clock_bias"
                csv_file.write(header + '\n')
                if self.__data.file_standard == "RINEX":
                    data = self.__data.get_satellite_data(sat)
                elif self.__data.file_standard == "SP3":
                    data = self.__data.get_satellite_data(sat, data_type=self.sp3_data_type)
                else:
                    data = []
                for epoch, clock_bias in data:
                    # TODO: omit naming convention; unify clock bias in RINEX and SP3 to bias or clock (or ...?)
                    bias = clock_bias.bias if self.__data.file_standard == "RINEX" else clock_bias.clock
                    csv_file.write("{}{}{}\n".format(epoch, self.csv_sep, float(bias) * scale))

    def csv_to_data(self):
        # clock_file = GnssClockData(dir_name="/home/pawel",
        #                            file_standard="SP3")
        #
        # keys = clock_file.data.keys()
        #
        # sat = "G23"
        # epochs = ['1949.138392857143', '1949.139880952381', '1949.141369047619']
        # clock = ['0.1', '0.2', '-0.3']
        #
        # clock_file.data[keys[0]].update_clock(sat, epochs, clock, 'Predicted')
        #
        # clock_file.data[keys[0]].save("/home/pawel/test.sp3")

        # TODO: implement above operations using csv files
        pass
