# DESCRIPTION
# File presents comparison of IGS Ultra-rapid Half-observed/predicted products and reading of the Sp3 file format
#
# Our input data is the Sp3 file format. It consists of 48h of data (15 minute sampling interval).
# Half of clock biases come from observed data and the other half is predicted
#
# Data to read and analyze is as follows:
#
# (2022 0)                 (2022 1)                 (2022 2)                 (2022 3)               (epoch)
#    |------------------------|------------------------|------------------------|--------------->   timeline
#
#          24 h observed            24 h predicted
#    |------------------------|------------------------|                                            igu20220_00.sp3
#
#                                   24 h observed            24 h predicted
#                             |------------------------|------------------------|                   igu20221_00.sp3

from core.gnss.gnss_clock_data import GnssClockData
import matplotlib.pyplot as plt
import numpy as np

SCALE = 10.0 ** 3  # scale micros to ns


def main():
    sat_number = 'G05'

    # --- Collect data
    # Get prediction data (as a prediction data I understand satellite clock corrections from predictions,
    # the same for observation data)
    print("Let's read prediction part...")
    pred_data = GnssClockData(dir_name="clock_data/sp3_sample_1",
                              file_standard="SP3")
    pred_sat_data = pred_data.get_satellite_data(sat_number, data_type='Predicted')
    epochs_pred = []
    clock_biases_pred = []
    for epoch, clock_bias in pred_sat_data:
        epochs_pred.append(epoch)
        clock_biases_pred.append(float(clock_bias.clock) * SCALE)

    # Get observation data
    print("\nLet's read observation part...")
    obs_data = GnssClockData(dir_name="clock_data/sp3_sample_2",
                             file_standard="SP3")
    obs_sat_data = obs_data.get_satellite_data(sat_number, data_type='Observed')
    epochs_obs = []
    clock_biases_obs = []
    for epoch, clock_bias in obs_sat_data:
        epochs_obs.append(epoch)
        clock_biases_obs.append(float(clock_bias.clock) * SCALE)

    # ---- Verification
    # Both arrays have to contain the same number of corrections...
    if len(clock_biases_pred) != len(clock_biases_obs):
        print("ERROR: number of clock corrections in predicted and observed parts is different!")
        return

    # and have to have exactly the same epochs!
    for i in range(len(epochs_pred)):
        if epochs_pred[i] != epochs_obs[i]:
            print("ERROR: one of the epochs is different!")
            return

    # --- Visualize results
    # Compare two estimations
    plt.plot([i for i in range(len(clock_biases_obs))], clock_biases_obs, 'o', label="Observed part")
    plt.plot([i for i in range(len(clock_biases_pred))], clock_biases_pred, 'x', label="Predicted part")
    plt.title('Satellite {} and its IGS Ultra-rapid satellite clock corrections (biases)'.format(sat_number))
    plt.ylabel('Clock correction [ns]')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    difference = np.array(clock_biases_obs) - np.array(clock_biases_pred)
    print("\nMean difference = {0:.4f} [ns]".format(difference.mean()))

    # Show prediction error (!)
    plt.plot([i for i in range(len(difference))], difference)
    plt.title('IGS Ultra-rapid prediction error')
    plt.ylabel('Error [ns]')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    main()
