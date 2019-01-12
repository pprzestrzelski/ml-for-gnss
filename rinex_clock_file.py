

class RinexClockFile:
    """
    Based on ftp://igs.org/pub/data/format/rinex_clock300.txt
    """
    def __init__(self, file_path):
        self.path_to_file = file_path
        self.file_name = file_path      # FIXME: it is not true for absolute path!
        # RINEX file consists of two parts: header and data (e.g. obs, nav or ... clock:))
        self.header = None
        self.data = None
        self.__read_file(file_path)

    def __read_file(self, path):
        with open(path, 'r') as f:
            self.__read_header(f)
            if self.__valid_clock_file():
                self.__read_data(f)

    def __read_header(self, f):
        header = []
        for line in f:
            header.append(line)
            if line.strip() == 'END OF HEADER':
                self.header = RinexClockHeader(header)
                break

    def __valid_clock_file(self):
        if self.header.rinex_file_version() < 3.0 and self.header.rinex_file_type() is not 'C':
            print("WARNING: this file", self.file_name, "is not RINEX >= 3.00 and of a clock type!")
            print("WARNING: RINEX clock data won't be read...")
            return False
        else:
            return True

    def __read_data(self, f):
        # FIXME: only one line of data per sat/rec is assumed! In fact it may be more than one
        self.data = RinexClockData()
        line = f.readline()
        if line is None or line == "":
            print("WARNING: no data in the RINEX data section!")
            return

        date, clock_record = self.__get_clock_data(line)
        data_block = RinexClockDataBlock(date)
        data_block.add_record(clock_record.get_name(), clock_record)
        self.data.add(data_block)

        prev_epoch = date.get_epoch()
        for line in f:
            date, clock_record = self.__get_clock_data(line)
            # data_block = RinexClockDataBlock(date)
            data_block.add_record(clock_record.get_name(), clock_record)
            epoch = date.get_epoch()
            if prev_epoch < epoch:
                data_block = RinexClockDataBlock(date)
                self.data.add(data_block)
                prev_epoch = epoch
            data_block.add_record(clock_record.get_name(), clock_record)

    def __get_clock_data(self, line):
        arr = line.split()
        data_type = arr[0]
        name = arr[1]
        date = RinexDate(self.header.get_gps_week(), self.header.get_gps_day(),
                         arr[2], arr[3], arr[4], arr[5], arr[6], arr[7])
        data_count = int(arr[8])
        clock_record = None     # can remove if if-statement will be fully implemented
        if data_count == 2:
            clock_record = RinexClockDataRecord(data_type, name, arr[9], arr[10])
        elif data_count == 4:
            pass   # TODO: implement
        elif data_count == 6:
            pass   # TODO: implement
        else:
            print("ERROR: not handled number of values ({}) to follow in the RINEX data section:".format(data_count))

        return date, clock_record

    def first_epoch(self):
        return self.data.get(0).get_epoch()

    def count_epochs(self):
        return self.data.size()

    def file_type(self):
        return self.header.rinex_file_type()

    def file_version(self):
        return self.header.rinex_file_version()

    def get_data(self, sat):
        data = []
        for i in range(self.data.size()):
            block = self.data.get(i)
            if block.contains(sat):
                data.append((block.get_epoch(), block.get_record(sat)))
            else:
                print("WARNING: missing data for {} on {}".format(sat, block.get_readable_epoch()))
        return data


class RinexClockHeader:
    def __init__(self, array_data):
        self.header_data = array_data
        self.rinex_version = 0
        self.rinex_type = 'X'          # just an arbitrary value, X as unknown
        self.__read_version_and_file_type()
        self.gps_week = "-1"      # -1 is just a dummy value
        self.gps_day = "-1"         # -1 is just a dummy value
        self.__read_gps_week_and_day()

    def __read_version_and_file_type(self):
        line = self.header_data[0]      # FIXME: version and clock does not have to be in the first line??!!
        if "RINEX VERSION / TYPE" in line:
            arr = line.split()
            self.rinex_version = float(arr[0])
            self.rinex_type = arr[1]
        else:
            print("WARNING: RINEX version/type is not in the first line!")

    def __read_gps_week_and_day(self):
        for line in self.header_data:
            if "GPS week:" in line:
                arr = line.split()
                for i in range(len(arr)):
                    if arr[i] == "GPS" and arr[i+1] == "week:":
                        self.gps_week = arr[i+2]
                    if arr[i] == "Day:":
                        self.gps_day = arr[i+1]
                return
        print("ERROR: RINEX file does not contain GPS week or GPS day")

    def rinex_file_version(self):
        return self.rinex_version

    def rinex_file_type(self):
        return self.rinex_type

    def get_gps_week(self):
        return self.gps_week

    def get_gps_day(self):
        return self.gps_day


class RinexClockData:
    def __init__(self):
        self.data_blocks = []   # each block is a new epoch

    def __len__(self):
        return len(self.data_blocks)

    def add(self, block):
        self.data_blocks.append(block)

    def get(self, index):
        return self.data_blocks[index]

    def size(self):
        return len(self.data_blocks)


class RinexClockDataBlock:
    def __init__(self, rinex_date):
        self.date = rinex_date
        self.epoch = self.date.get_epoch()
        self.records = {}

    def size(self):
        return len(self.records)

    def add_record(self, name, record):
        self.records[name] = record

    def get_record(self, name):
        return self.records[name]   # TODO: assumes 'name' record exists. Should throw error otherwise?

    def contains(self, name):
        if name in self.records:
            return True
        else:
            return False

    def get_date(self):
        return self.date

    def get_epoch(self):
        return self.epoch

    def get_readable_epoch(self):
        return self.date.get_readable_epoch()


class RinexDate:
    def __init__(self, gps_week, gps_day, year, month, day, hour, minute, second):
        self.gps_week = gps_week
        self.gps_day = gps_day      # gps day != calendar day (self.day)
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def get_epoch(self):
        return float(self.gps_week) + float(self.gps_day) / 7 + \
               float(self.hour) / 7 / 24 + float(self.minute) / 7 / 24 / 60 + \
               float(self.second) / 7 / 24 / 60 / 3600

    def get_readable_epoch(self):
        return "{}-{}-{} {}:{}:{}".format(self.year, self.month, self.day, self.hour, self.minute, self.second)


class RinexClockDataRecord:
    def __init__(self, data_type, rec_sat_name, bias, bias_sigma=None,
                 rate=None, rate_sigma=None, acceleration=None, acceleration_sigma=None):
        self.type = data_type               # AR, AS, CR, DR or MS
        self.rec_sat_name = rec_sat_name    # receiver or satellite name, 4 or 3 char designator, respectively
        self.bias = bias                # seconds
        self.bias_sigma = bias_sigma    # seconds
        self.rate = rate                # dimensionless
        self.rate_sigma = rate_sigma    # dimensionless
        self.acceleration = acceleration                # per second
        self.acceleration_sigma = acceleration_sigma    # per second

    def get_name(self):
        return self.rec_sat_name

    def get_bias(self):
        return self.bias

    def get_bias_sigma(self):
        return self.bias_sigma
