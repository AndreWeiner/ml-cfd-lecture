"""
    This file has function to calculate rewards and returns from the data collected by trajectories.
    The data from trajectory is fetched by "read_trajectory_data.py"
    This file has also a function to computes GAEs.

    called in : replay_buffer.py
"""

import numpy as np


def calculate_rewards_returns(r_1, r_2, r_3, r_4, coeff_data, gamma, theta_, d_theta):
    """
    To compute the rewards and returns from the sampled trajectory.
    Args:
        r_1: coefficient of reward function
        r_2: coefficient of reward function
        r_3: coefficient for reward function
        r_4: coefficient for reward function
        coeff_data: csv dataframe from trajectory containing data of c_d and c_l
        gamma: discount factor

    Returns: rewards array and return array

    """
    # get c_d and c_l value from coeff dataframe
    c_d = coeff_data.c_d.values
    c_l = coeff_data.c_l.values

    # reward function to compute rewards
    rewards = r_1 - (c_d + r_2 * abs(c_l) + r_3 * theta_ + r_4 * d_theta)

    # length of an reward array -> length of a trajectory
    T = len(rewards)

    # discout factor to compute return at the end of trajectories.
    discounts = np.logspace(0, T, num=T, base=gamma, endpoint=False)

    # return -> sum of discount factor x rewards
    returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])

    return rewards, returns


def calculate_gaes(values_pi, rewards, gamma, lambda_):
    """
    To compute GAEs from rewards and value in order to determine how good was the taken action.(Criticism -> critic)
    Args:
        values_pi: state values array from the value model
        rewards: rewards array
        gamma: discount factor
        lambda_: coeff of TD-lamda method

    Returns: GAE array

    """

    # length of a trajectory
    n_T = rewards.shape[1]

    # no of a trajectory
    n_traj = rewards.shape[0]

    # initialization of gae array
    gae_buffer = np.zeros((n_traj, n_T-1))

    # labmda factor for TD_lambda method -> computed with gamma*lambda
    # GAE equation -> lamda( R + gamma * V_pi(t+1) + V_pi(t)
    lambda_discounts = np.logspace(0, n_T - 1, num=n_T - 1, base=gamma * lambda_, endpoint=False, dtype=np.float128)

    for i in range(n_traj-1):
        # getting V_pi and R for specific trajectory
        p_values_pi = values_pi[i]
        r_rewards = rewards[i]
        # GAE equation -> lamda( R + gamma * V_pi(t+1) + V_pi(t)
        deltas = r_rewards[:-1] + gamma * p_values_pi[1:] - p_values_pi[:-1]
        gaes = np.array([np.sum(lambda_discounts[:n_T - 1 - t] * deltas[t:]) for t in range(n_T - 1)])

        # filling GAE buffer for GAE value of specific trajectory.
        gae_buffer[i] = gaes

    return gae_buffer
