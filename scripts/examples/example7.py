from core.utils.clock_data_converter import ClockDataConverter
from core.gnss.gnss_clock_data import GnssClockData


def main():
    # 1. Prepare csv files with clock data for manipulation
    clock_data = GnssClockData(dir_name="clock_data")
    converter = ClockDataConverter()
    converter.add_data(clock_data)
    converter.data_to_csv()

    # 2. Manipulate csv files in the "conversions/conversionXXXXXXXX_XXXXXX/raw_csv"
    #       read to Pandas
    #       predict
    #       etc.

    # 3. Save results to "conversions/conversionXXXXXXXX_XXXXXX/modified_csv"
    #    Get above directory using: converter.dir_modified_csv

    # 4. Inject modified values of clock biases to a GnssClockData file
    converter.csv_to_data()

    # 5. Save new RINEX/SP3 to files using a GnssClockData object, e.g.:
    keys = clock_data.data.keys()
    clock_data.data[keys[0]].save("/home/your_home_dir/output_gnss_clock_data.sp3_or_clk")


if __name__ == "__main__":
    main()
