from scripts.core.utils.gnss_time import utc_to_gps_week_day
from scripts.core.gnss.gnss_date import GnssDate


class Sp3File:
    """
    Based on ftp://igs.org/pub/data/format/sp3c.txt
    """
    def __init__(self, file_path):
        self.path_to_file = file_path
        self.file_name = file_path      # FIXME: does not have to be true
        self.header = None
        self.data = []  # Sp3DataBLock(s)
        self.__read_file(file_path)

    def __read_file(self, path):
        with open(path, 'r') as f:
            self.__read_header(f)
            if self.__valid_clock_file():
                self.__read_data(f)

    def __read_header(self, f):
        header = []
        i = 1
        while i <= 22:
            line = f.readline()
            header.append(line)
            i += 1
        self.header = Sp3Header(header)

    def __valid_clock_file(self):
        if self.file_name.endswith(".sp3"):
            return True
        return False

    def __read_data(self, f):
        line = f.readline()
        if line is None or line == "":
            print("ERROR: next line after header is not data related!")
            return

        if line[0] != "*":
            print("ERROR: next line after header is not gnss date")
            return

        gnss_date = Sp3File.gnss_date_from_sp3(line)
        data_block = Sp3DataBlock(gnss_date)
        line = f.readline()

        while 'EOF' not in line:
            if line[0] == '*':
                self.data.append(data_block)
                gnss_date = Sp3File.gnss_date_from_sp3(line)
                data_block = Sp3DataBlock(gnss_date)
            elif line[0] == 'P':
                record = Sp3PositionRecord(line)
                data_block.records[record.sat] = record
            elif line[0] == 'V':
                continue    # TODO: implement sp3 velocity record
            else:
                continue    # TODO: implement sp3 correlation records

            line = f.readline()

        self.data.append(data_block)

    @staticmethod
    def gnss_date_from_sp3(str_line):
        year, month, day = str_line[3:7], str_line[8:10], str_line[11:13]
        hour, minute = str_line[14:16], str_line[17:19]
        str_second = str_line[20:31]
        second_and_parts = str_line[20:31].split('.')
        second = second_and_parts[0] + '.' + second_and_parts[1][:6]    # we have to trim to 6 digits after comma
        utc_time = year.strip() + "-" + month.strip() + "-" + day.strip() + " " + \
            hour.strip() + ":" + minute.strip() + ":" + second.strip()
        utc_format = "%Y-%m-%d %H:%M:%S.%f"
        gps_week, gps_day = utc_to_gps_week_day(utc_time, utc_format)

        return GnssDate(float(gps_week), float(gps_day), year, month, day, hour, minute, str_second)

    def first_epoch(self):
        return self.data[0].epoch

    def file_type(self):
        return self.header.file_data_type

    def file_version(self):
        return self.header.file_version

    def get_data(self, sat, data_to_read):
        """
        :param sat: GNSS satellite name, e.g. 'G01' is the GPS satellite #01
        :param data_to_read: 'Observed', 'Predicted' or 'Both'
        """
        data = []
        for i in range(len(self.data)):
            if sat in self.data[i].records:
                record = self.data[i].records[sat]
                if record.clock_pred_flag == ' ' and data_to_read == 'Observed':
                    data.append((self.data[i].epoch, record))
                elif record.clock_pred_flag == 'P' and data_to_read == 'Predicted':
                    data.append((self.data[i].epoch, record))
                elif data_to_read == 'Both':
                    data.append((self.data[i].epoch, record))
        return data

    def save(self, file_name, prefix="", suffix=""):
        file = prefix + file_name + suffix
        with open(file, 'w') as f:
            f.writelines(self.header.header_data)

            for block in self.data:
                f.write(block.sp3_time_line())
                f.write('\n')

                for sat in block.records.keys():
                    f.write(block.records[sat].sp3_record())
                    f.write('\n')
            f.write('EOF')
            f.write('\n')

    def update_clock(self, sat, epochs, clock_data, data_type):
        """
        Method updates satellite clock data within given epochs and chosen data type (predicted or observed).
        It assumes pairs of epochs and clock data sorted ascending as an input.
        :param sat: GNSS satellite name, e.g. 'G01' is the GPS satellite #01
        :param epochs: array of epochs [floats or strings]
        :param clock_data: array of clock biases as [floats or strings]
        :param data_type: 'Predicted' or 'Observed' data type to update
        """
        i = 0
        for j in range(len(self.data)):
            if self.data[j].epoch == float(epochs[0]):
                break

            i += 1
            if i == len(self.data):
                print("ERROR: could't find requested data")
                return

        for k in range(len(epochs)):
            record = self.data[i].records[sat]

            if record.symbol == 'P' and data_type == 'Predicted' \
                    or record.symbol == ' ' and data_type == 'Observed':
                record.clock = clock_data[k]
            else:
                print("ERROR: found epoch but data type to update is not the right one")
                return

            i += 1


class Sp3Header:
    def __init__(self, array_data):
        self.header_data = array_data
        self.file_version = self.__read_file_version()  # a, b or c
        self.file_data_type = self.__read_file_data_type()  # P or V (position or velocity data type respectively)
        self.gps_week = "-1"
        self.gps_day = "-1"
        self.__read_gps_week_and_day()
        self.base_for_pos_vel_acc = None
        self.base_for_clk_rate_acc = None
        self.__read_acc_bases()

    def __read_file_version(self):
        return self.header_data[0][1]

    def __read_file_data_type(self):
        return self.header_data[0][2]

    def __read_gps_week_and_day(self):
        str_date = self.header_data[0].split()
        year, month, day = str(str_date[0][3:]), str_date[1], str_date[2]
        hour, minute = str_date[3], str_date[4]
        second_and_parts = str_date[5].split('.')
        second = second_and_parts[0] + '.' + second_and_parts[1][:6]    # we have to trim to 6 digits after comma
        utc_time = year + "-" + month + "-" + day + " " + hour + ":" + minute + ":" + second
        utc_format = "%Y-%m-%d %H:%M:%S.%f"
        self.gps_week, self.gps_day = utc_to_gps_week_day(utc_time, utc_format)

    def __read_acc_bases(self):
        self.base_for_pos_vel_acc = self.header_data[14][3:13]
        self.base_for_clk_rate_acc = self.header_data[14][14:26]


class Sp3DataBlock:
    """
    Sp3DataBlock may consist of Sp3 position, velocity or correlation records
    """
    def __init__(self, gnss_date):
        self.date = gnss_date
        self.epoch = self.date.get_epoch()
        self.records = {}  # Sp3Record(s)

    def sp3_time_line(self):
        blank = ' '
        return '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(
            '* ', blank, self.date.year, blank, self.date.month, blank, self.date.day,
            blank, self.date.hour, blank, self.date.minute, blank, self.date.second)


class Sp3PositionRecord:
    def __init__(self, string_line):
        """
        Gets data as a string and slices it according to the sp3c file
        specification (ftp://igs.org/pub/data/format/sp3c.txt)
        (no, so may blank spaces is not mistake).

        Values of (X, Y, Z) and Clock standard deviations given here (in the sp3 position record) are only
        exponents for accuracy (precision). Accuracy (precision) values can be obtained according to:
            acc = b ** n
            where: acc - accuracy (precision) in mm or psec (for pos or clk respectively)
                   b - is a base from Sp3 Header (usually 1.25 and 1.05 for pos and clk respectively)
                   n - pos/clock sdev (empty, i.e. ' ', sdev value in the Sp3 position record means std is unknown)

        :param string_line: e.g. "PG01 -16355.997140  -2581.834262  20666.202859    -96.730418 11  7  7 215       "
        """

        self.symbol = string_line[0]    # 'P'
        self.sat = string_line[1:4]     # e.g. 'G01'
        self.x = string_line[4:18]        # [km]
        self.y = string_line[18:32]       # [km]
        self.z = string_line[32:46]       # [km]
        self.clock = string_line[46:60]   # [micro sec]
        self.x_sdev = string_line[61:63]  # [mm] or ' ' (means std is unknown)
        self.y_sdev = string_line[64:66]  # [mm] or ' ' (means std is unknown)
        self.z_sdev = string_line[67:69]  # [mm] or ' ' (means std is unknown)
        self.clock_sdev = string_line[70:73]    # [pico sec] or ' ' (means std is unknown)
        self.clock_event_flag = string_line[74]    # ' ' or 'E'
        self.clock_pred_flag = string_line[75]     # ' ' or 'P'
        self.maneuver_flag = string_line[78]       # ' ' or 'M'
        self.orbit_pred_flag = string_line[79]     # ' ' or 'P'

    def sp3_record(self):
        blank = ' '
        return '{:1s}{:3s}{:14.6f}{:14.6f}{:14.6f}{:14.6f}{}{:2s}{}{:2s}{}{:2s}{}{:3s}{}{:1s}{:1s}{}{}{:1s}{:1s}'\
            .format(self.symbol, self.sat, float(self.x), float(self.y), float(self.z), float(self.clock), blank,
                    self.x_sdev, blank, self.y_sdev, blank, self.z_sdev, blank, self.clock_sdev,
                    blank, self.clock_event_flag, self.clock_pred_flag, blank, blank, self.maneuver_flag,
                    self.orbit_pred_flag)


class Sp3VelocityRecord:
    pass


class Sp3PosVelCorrelationRecord:
    pass


# ROC => Rate-of-change
class Sp3VelClockROCCorrelationRecord:
    pass
