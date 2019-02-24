from core.utils.gnss_time import utc_to_gps_week_day
from core.gnss.gnss_date import GnssDate


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
        second_and_parts = str_line[20:31].split('.')
        second = second_and_parts[0] + '.' + second_and_parts[1][:6]    # we have to trim to 6 digits after comma
        utc_time = year.strip() + "-" + month.strip() + "-" + day.strip() + " " + \
            hour.strip() + ":" + minute.strip() + ":" + second.strip()
        utc_format = "%Y-%m-%d %H:%M:%S.%f"
        gps_week, gps_day = utc_to_gps_week_day(utc_time, utc_format)

        return GnssDate(float(gps_week), float(gps_day), year, month, day, hour, minute, second)

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


class Sp3Header:
    def __init__(self, array_data):
        self.header_data = array_data
        self.file_version = self.__read_file_version()  # a, b or c
        self.file_data_type = self.__read_file_data_type()  # P or V (position or velocity data type respectively)
        self.gps_week = "-1"
        self.gps_day = "-1"
        self.__read_gps_week_and_day()

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


class Sp3DataBlock:
    """
    Sp3DataBlock may consist of Sp3 position, velocity or correlation records
    """
    def __init__(self, gnss_date):
        self.date = gnss_date
        self.epoch = self.date.get_epoch()
        self.records = {}  # Sp3Record(s)


class Sp3PositionRecord:
    def __init__(self, string_line):
        """
        Gets data as a string and slices it according to the sp3c file
        specification (ftp://igs.org/pub/data/format/sp3c.txt)
        (no, so may blank spaces is not mistake).
        :param string_line: e.g. "PG01 -16355.997140  -2581.834262  20666.202859    -96.730418 11  7  7 215       "
        """
        self.symbol = string_line[0]    # 'P'
        self.sat = string_line[1:4]     # e.g. 'G01'
        self.x = string_line[4:18]        # [km]
        self.y = string_line[18:32]       # [km]
        self.z = string_line[32:46]       # [km]
        self.clock = string_line[46:60]   # [micro sec]
        self.x_sdev = string_line[61:63]  # [mm]
        self.y_sdev = string_line[64:66]  # [mm]
        self.z_sdev = string_line[67:69]  # [mm]
        self.clock_sdev = string_line[70:73]    # [pico sec]
        self.clock_event_flag = string_line[74]    # ' ' or 'E'
        self.clock_pred_flag = string_line[75]     # ' ' or 'P'
        self.maneuver_flag = string_line[78]       # ' ' or 'M'
        self.orbit_pred_flag = string_line[79]     # ' ' or 'P'


class Sp3VelocityRecord:
    pass


class Sp3PosVelCorrelationRecord:
    pass


# ROC => Rate-of-change
class Sp3VelClockROCCorrelationRecord:
    pass
