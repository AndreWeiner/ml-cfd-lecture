"""
    This file has function to evaluated the score at the end of iteration of main PPO loop
    This file also has a function to update the traced model in base_case folder,
        containing OpenFOAM file. This traced model is used in OpenFOAM simulations.

    called in : ppo.py
"""

import os
import numpy as np
import torch


def evaluate_score(rewards, sample):
    """
    This Funciton is to calculate the evaluation score at the end of main PPO iteration.
    Args:
        rewards: reward array
        sample: iteration number

    Returns: evaluation score array of arrays : rewards_mean -> mean of rewards array
                                               rewards_std -> std of rewards array
                                               mean_10_rewards -> mean of last 10 rewards (for moving average)
                                               std_10_rewards -> std of last 10 rewards (for moving average)
                                               mean_100_rewards -> mean of last 100 rewards (for moving average)
                                               std_100_rewards -> std of last 100 rewards (for moving average)

    """
    # mean and std of rewards
    reward_mean = rewards.mean()
    reward_std = rewards.std()

    # mean and std of last 10 values of rewards array (for moving average)
    mean_10_rewards = rewards[:, -10:].mean()
    std_10_rewards = rewards[:, -10:].std()

    # mean and std of last 10 values of rewards array (for moving average)
    mean_100_rewards = rewards[:, -100:].mean()
    std_100_rewards = rewards[:, -100:].std()

    # array containing rewards mean_s and std_s
    evaluations = np.array([reward_mean, reward_std,
                            mean_10_rewards, std_10_rewards,
                            mean_100_rewards, std_100_rewards])

    # to save the list in Disc memory for later post processing -> 'results/evaluation/evaluation_0.npy'
    path = 'results/evaluation'
    os.makedirs(path, exist_ok=True)
    np.save(path + f'/evaluations_{sample}.npy', evaluations)

    return evaluations


def model_trace_update(model, sample):
    """
    This function is update traced policy model for to use by OpenFOAM to rotate the cylinder
    at certain angular velocity at every 20 time steps, sampled by model

    Args:
        model: policy model
        sample: iteration number

    Returns: saving traced model in OpenFOAM dirs

    """

    path = "results/models"
    base_path = "env/base_case/agentRotatingWallVelocity"
    os.makedirs(path, exist_ok=True)

    # tracing policy model
    traced_cell = torch.jit.script(model)

    # saving model to disc memory for post processing
    traced_cell.save(path + f"/policy_{sample}.pt")

    # remove previous model, in order to use the latest
    os.system(f"rm {base_path}/policy.pt")

    # saving model to openFOAM dirs
    os.system(f"cp {path}/policy_{sample}.pt {base_path}/policy.pt")


def saving_value_model(model, sample):
    """
		This function saves value model at the end of every epoch.
	"""
    path = "results/value_models"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), path + f"/value_{sample}.pt")
