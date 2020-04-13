class GnssDate:
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
