"""
    This file has function containg main PPO algorithm

    called in : main.py
"""

import time

from reply_buffer import *
from eval_score_and_trace_update import *


def train_model(value_model,
                policy_model,
                env,
                policy_optimizer,
                policy_optimization_epochs,
                policy_sample_ratio,
                policy_clip_range,
                policy_model_max_grad_norm,
                policy_stopping_kl,
                entropy_loss_weight,
                value_optimization_epochs,
                value_optimizer,
                value_sample_ratio,
                value_clip_range,
                value_model_max_grad_norm,
                value_stopping_mse,
                gamma,
                lambda_,
                r_1,
                r_2,
                r_3,
                r_4,
                sample,
                n_sensor,
                EPS,
                evaluation_score,
                action_bounds):
    """

    Args:
        value_model: value model instance
        policy_model: polica model instance
        env: env class instance
        policy_optimizer: policy optimizer -> adam
        policy_optimization_epochs: no of epoch for policy training
        policy_sample_ratio: no of trajectory to take for training
        policy_clip_range: clipping parameter of policy loss
        policy_model_max_grad_norm: maximum norm tolerance of policy optimization
        policy_stopping_kl: tolerance for training of policy net
        entropy_loss_weight: factor for entropy loss
        value_optimization_epochs: no of epochs for value model
        value_optimizer: value optimizer -> adam
        value_sample_ratio: no of trajectory to take for training
        value_clip_range: clipping parameter of value model loss
        value_model_max_grad_norm: maximum norm tolerance of value optimization
        value_stopping_mse: tolerance for trainig of value net
        gamma: discount factor
        lambda_: TD_lambda method factor
        r_1: coefficient for reward function
        r_2: coefficient for reward function
        r_3 coefficient for reward function
        r_4: coefficient for reward function
        sample: number of ppo iteration
        n_sensor: no of patches at the surface of cylinder
        EPS: Tolerance for std
        evaluation_score: evaluation score for post processing results
        action_bounds: min and max omega value

    Returns: trajectory running time and time to run one iteration of ppo main algorithm

    """
    # starting time to calculate time of trajectory run and each iteration of main algorithm
    # getting variable for ppo algorithm from reply_buffer.py
    traj_start_time = time.perf_counter()
    states, actions, rewards, returns, logpas = fill_buffer(env, sample, n_sensor, gamma, r_1, r_2, r_3, r_4, action_bounds)
    traj_time = (time.perf_counter() - traj_start_time)

    # get V_pi for the state values
    values_pi = value_model(torch.from_numpy(states)).squeeze().detach().numpy()

    # compute GAEs of taken actions and the obtained rewards
    gaes = calculate_gaes(values_pi, rewards, gamma, lambda_)

    gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)

    # no of trajectories
    n_samples = len(actions)

    for q in range(policy_optimization_epochs):

        # ramdom selection of trajectories from the reply buffer
        batch_size = int(policy_sample_ratio * n_samples)
        batch_idxs = np.random.choice(n_samples, batch_size, replace=False)

        # get the data for chosen random  selected trajectory
        states_batch = states[batch_idxs]
        actions_batch = actions[batch_idxs]
        gaes_batch = gaes[batch_idxs]
        logpas_batch = logpas[batch_idxs]

        # log probabilities and entropy for randomly chosen trajectory
        logpas_pred, entropies_pred = policy_model.get_predictions(states_batch[:, :-1, :], actions_batch)

        # ratio of log probability to calculate the loss and clipping of policy loss
        # compute entropy loss
        ratios = (logpas_pred - torch.from_numpy(logpas_batch)).exp()
        pi_obj = torch.from_numpy(gaes_batch) * ratios
        pi_obj_clipped = torch.from_numpy(gaes_batch) * ratios.clamp(1.0 - policy_clip_range, 1.0 + policy_clip_range)
        policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
        entropy_loss = -entropies_pred.mean() * entropy_loss_weight

        # total loss (entropy loss + policy loss) back propagation
        policy_optimizer.zero_grad()
        (policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), policy_model_max_grad_norm)
        policy_optimizer.step()

        # checking for optimization in range of tolerance
        with torch.no_grad():
            logpas_pred_all, _ = policy_model.get_predictions(states[:, :-1, :], actions)
            kl = (torch.from_numpy(logpas) - logpas_pred_all).mean()
            if kl.item() > policy_stopping_kl:
                print(f'Difference between old and new policy larger than tolerance: {q} and {kl.item()}')
                break

    model_trace_update(policy_model, sample)

    for q in range(value_optimization_epochs):

        # ramdom selection of trajectories from the reply buffer
        batch_size = int(value_sample_ratio * n_samples)
        batch_idxs = np.random.choice(n_samples, batch_size, replace=False)

        # get the data for chosen random  selected trajectory
        states_batch = states[batch_idxs]
        returns_batch = returns[batch_idxs]
        values_batch = values_pi[batch_idxs]

        # getting V_pi for randomly selected trajectories in reply buffer
        values_pred = value_model(torch.from_numpy(states_batch)).squeeze()
        values_pred_clipped = torch.from_numpy(values_batch) + (values_pred - torch.from_numpy(values_batch)).clamp(
            -value_clip_range, value_clip_range)

        # critic loss
        v_loss = (torch.from_numpy(returns_batch) - values_pred).pow(2)
        v_loss_clipped = (torch.from_numpy(returns_batch) - values_pred_clipped).pow(2)
        value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

        # critic loss optimization
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), value_model_max_grad_norm)
        value_optimizer.step()

        # checking for optimization in range of tolerance
        with torch.no_grad():
            values_pred_all = value_model(torch.from_numpy(states)).squeeze()
            mse = (torch.from_numpy(values_pi) - values_pred_all).pow(2).mul(0.5).mean()
            if mse.item() > value_stopping_mse:
                print(f'MSE of value network larger than tolerance: {q}, {mse.item()}')
                break

    # saving value model
    saving_value_model(value_model, sample)

    # computation time to complete one iteration of main PPO algorithm
    epoch_time = (time.perf_counter() - traj_start_time)

    # evaluation score at the end of iteration
    score = evaluate_score(rewards, sample)
    evaluation_score.append(score)

    return traj_time, epoch_time
