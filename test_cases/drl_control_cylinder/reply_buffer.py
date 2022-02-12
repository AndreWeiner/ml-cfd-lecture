"""
    This file has function to sample the trajectories and extract data,
    as states, rewards, actions, logporbs and rewards, from it.

    called in :  ppo.py
"""

from read_trajectory_data import *
from cal_R_gaes import *
from check_traj import *
from env_local import *


def fill_buffer(env, sample, n_sensor, gamma, r_1, r_2, r_3, r_4, action_bounds):
    """
    This function is to sample trajectory and get states, actions, probabilities, rewards,
    and to calculate returns.

    Args:
        env: instance of environment class for sampling trajectories
        sample: number of iteration
        n_sensor: no of patches at the surface of cylinder
        gamma: discount factor
        r_1: coefficient for reward function
        r_2: coefficient for reward function
        r_3: coefficient for reward function
        r_4: coefficient for reward function
        action_bounds: min and max omega value

    Returns: arrays of, states, actions, rewards, returns, probabilities

    """

    # to sample the trajecties
    env.sample_trajectories(sample, action_bounds)
    
    # check the trajectory to be completed
    check_trajectories(sample)

    traj_files = glob(f'./Data/sample_{sample}' + "/*/")

    # To check if the trajectories is sampled
    n_traj = len(traj_files)
    assert n_traj > 0

    # To extract the length of trajectory
    t_traj = pd.read_csv(traj_files[0] + "trajectory.csv", sep=",", header=0)
    n_T = len(t_traj.t.values)

    # due to delayed starting behaviour the time steps set to explicit -> length of trajectory
    # choose shortest available trajectory to ensure the same length
    for i, files in enumerate(traj_files):
        # To extract the length of trajectory
        t_traj = pd.read_csv(traj_files[i] + "trajectory.csv", sep=",", header=0)
        n_T_temp = len(t_traj.t.values)
        if n_T_temp < n_T:
            n_T = n_T_temp

    # buffer initialization
    state_buffer = np.zeros((n_traj, n_T, n_sensor))
    action_buffer = np.zeros((n_traj, n_T - 1))
    reward_buffer = np.zeros((n_traj, n_T))
    return_buffer = np.zeros((n_traj, n_T))
    log_prob_buffer = np.zeros((n_traj, n_T - 1))

    for i, files in enumerate(traj_files):
        # get the dataframe from the trajectories
        coeff_data, trajectory_data, p_at_faces = read_data_from_trajectory(files, n_sensor)

        # sometimes off by 1 this fixes that
        n_T_2 = len(trajectory_data.t.values)
        if n_T_2 > n_T:
            trajectory_data = trajectory_data[:n_T]

        # state values from data frame
        states = trajectory_data[p_at_faces].values

        # action values from data frame
        actions_ = trajectory_data.omega.values
        actions = actions_[:-1]
        
        # rotation rate
        theta_ = trajectory_data.theta_sum.values
        d_theta = trajectory_data.dt_theta_sum.values

        # rewards and returns from cal_R_gaes.py -> calculate_rewards_returns
        rewards, returns = calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data, gamma, theta_, d_theta)

        # log_probs from data frame
        log_probs_ = trajectory_data.log_p.values
        log_probs = log_probs_[:-1]

        # appending values in buffer
        state_buffer[i] = states[:n_T, :]
        action_buffer[i] = actions[:n_T-1]
        reward_buffer[i] = rewards[:n_T]
        return_buffer[i] = returns[:n_T]
        log_prob_buffer[i] = log_probs[:n_T-1]

    return state_buffer, action_buffer, reward_buffer, return_buffer, log_prob_buffer
