"""
    This file to read the data from the sampled trajectories.

    called in : replay_buffer.py
"""

import pandas as pd
import os


def read_data_from_trajectory(traj_files, n_sensor):
    """
    This function is to read all the values from trajectory.
    meaningly states(pressure values), action taken(angular velocity), change in rotation,
    change in rotation per time, entropy of distribution, log probability of sampled action.
    It also reads the post processing coefficients, in order to calculate rewards.

    Args:
        traj_files: trajectory location relative to main ppo.py or env.py file

    Returns: Two dataframes.
             1) from trajectory.csv :  ["t", "omega", "omega_mean", "omega_log_std", "alpha", "beta", "log_p", "entropy", "theta_sum", "dt_theta_sum"]
             2) from postProcessing/forces/0/coefficient.dat" : ["t", "c_d", "c_l"]
    """

    # for debug
    assert traj_files, 'trajectory file not found, traj_files empty'

    # read trajectory data
    # number of cell faces forming the cylinder patch
    n_faces = n_sensor
    names = ["t", "omega", "omega_mean", "omega_log_std", "alpha", "beta", "log_p", "entropy", "theta_sum", "dt_theta_sum"]
    p_names = ["p{:d}".format(i) for i in range(n_faces)]

    trajectory = pd.read_csv(traj_files + "trajectory.csv", sep=",", names=names + p_names, header=0)

    # read force coefficients
    names_coeffs = ["col{:d}".format(i) for i in range(13)]

    # time, drag, and lift
    names_coeffs[0] = "t"
    names_coeffs[1] = "c_d"
    names_coeffs[3] = "c_l"
    keep = ["t", "c_d", "c_l"]

    file_path = traj_files + "coefficient.dat"
    coeffs = pd.read_csv(file_path, sep="\t", names=names_coeffs, usecols=keep, comment="#")

    # for some reason the function object's writeControls do not work properly; therefore,
    # we pick only every nth row from the dataframe; this number should be the same as for
    # the trajectory (specified in the boundary condition)
    pick_every = 20
    coeffs = coeffs[coeffs.index % pick_every == 0]

    # return values: Two Dataframes
    return coeffs, trajectory, p_names
