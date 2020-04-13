from datetime import datetime


def utc_to_gps_week_day(utc_time, utc_format):
    gps_zero_epoch = datetime.strptime("1980-01-06 00:00:00", "%Y-%m-%d %H:%M:%S")
    utc = datetime.strptime(utc_time, utc_format)

    diff = utc - gps_zero_epoch
    gps_week = diff / 7
    gps_day = utc.isoweekday() if utc.isoweekday() != 7 else 0

    return gps_week.days, gps_day
