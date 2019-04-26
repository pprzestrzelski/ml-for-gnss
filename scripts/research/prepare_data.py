from core.gnss.gnss_clock_data import GnssClockData
from core.utils.gnss_clock_stats import TrivialGnssClockStats
from core.utils.clock_data_converter import ClockDataConverter


# Collect train data and output it to csv (SP3 Observed data)
def collect_train_data():
    # Get and verify completeness
    clock_data = GnssClockData(dir_name="../../clock_data/research/train", file_standard="SP3")
    stats = TrivialGnssClockStats()
    stats.add_data(clock_data, 900)     # 15 min recording interval
    stats.print_missing()

    # Output to the CSV
    converter = ClockDataConverter(dir_name="train_data")
    converter.add_data(clock_data)
    converter.data_to_csv()


# Collect reference data and output it to csv (RINEX files)
def collect_reference_data():
    # 1. Reference for TRAIN data
    # Get and verify completeness
    cd_train = GnssClockData(dir_name="../../clock_data/research/reference/for_train_data", file_standard="RINEX")
    stats_train_data = TrivialGnssClockStats()
    stats_train_data.add_data(cd_train, 300)     # 5 min recording interval
    stats_train_data.print_missing()

    # Output to the CSV
    converter = ClockDataConverter(dir_name="reference_train_data")
    converter.add_data(cd_train)
    converter.data_to_csv()

    # 2. Reference for PREDICT data
    # Get and verify completeness
    cd_pred = GnssClockData(dir_name="../../clock_data/research/reference/for_predicted_data", file_standard="RINEX")
    stats_pred_data = TrivialGnssClockStats()
    stats_pred_data.add_data(cd_pred, 300)  # 5 min recording interval
    stats_pred_data.print_missing()

    # Output to the CSV
    converter = ClockDataConverter(dir_name="reference_predicted_data")
    converter.add_data(cd_pred)
    converter.data_to_csv()


# Collect predicted data from the IGU SP3 file (SP3 Prediction file)
def collect_igu_predicted():
    # Get and verify completeness
    clock_data = GnssClockData(dir_name="../../clock_data/research/predict", file_standard="SP3")
    stats = TrivialGnssClockStats()
    stats.add_data(clock_data, 900)  # 15 min recording interval
    stats.print_missing()

    # Output to the CSV
    converter = ClockDataConverter(dir_name="igu_predicted")
    converter.set_sp3_data_type('Predicted')
    converter.add_data(clock_data)
    converter.data_to_csv()


# Collect IGU observed data for prediction epochs (SP3 Observed file)
def collect_igu_observed():
    # Get and verify completeness
    clock_data = GnssClockData(dir_name="../../clock_data/research/igu_observed", file_standard="SP3")
    stats = TrivialGnssClockStats()
    stats.add_data(clock_data, 900)  # 15 min recording interval
    stats.print_missing()

    # Output to the CSV
    converter = ClockDataConverter(dir_name="igu_observed")
    converter.set_sp3_data_type('Observed')
    converter.add_data(clock_data)
    converter.data_to_csv()


def main():
    print("INFO: GNSS clock data will appear in the 'conversions' directory in the current work directory")
    # collect_train_data()
    # collect_reference_data()
    # collect_igu_predicted()
    collect_igu_observed()


if __name__ == "__main__":
    main()
