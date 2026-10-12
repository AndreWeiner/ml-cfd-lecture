# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ml-cfd
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Controlling the flow past a cylinder"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](cylinder_drl.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="cylinder_drl.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# In this notebook, we cover the following topics:
#
# 1. Flow control problems
# 2. Elements of reinforcement learning
# 3. SARSA and Q-learning
# 4. Elements of deep reinforcement learning
# 5. Proximal policy optimization

# %% [markdown]
# ## Flow control problems
#
# Flow control in the most general sense means that we manipulate a fluid flow to improve its behavior. There are two broad classes of flow control methods:
#
# 1. Passive flow control: the flow is manipulated by changing geometry, topology, or even the fluid itself; examples are winglets on aircraft wings or baffles in bended pipes for drag reduction
# 2. Active flow control: the flow is actuated by moving parts, blowing/suction, or heating/cooling; examples are moving wing flaps for gust mitigation or the injection of gas around a ships hull for drag reduction
#
# Active flow control methods may be further categorized into:
#
# 1. open-loop active flow control: in open-loop applications, no sensor input is required for the actuation; examples are constant or periodic motion, blowing, or heating
# 2. closed-loop active flow control: closed-loop control uses sensor inputs to determine the actuation; examples are flaps whose motion is controlled based on the input of pressure sensors
#
# Passive methods are the most straightforward means of flow control, but their effectiveness is limited, because they are typically optimized for one unique operation point. If the flow conditions change, passive methods may perform suboptimal. Active flow control methods can overcome this limitation but designing optimal control laws is challenging. For open-loop control, one may decide to rotate a component periodically
#
# ```
# https://github.com/mimoralea/gym-walk
# pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
# ```

# %%
from os import makedirs
from os.path import join
from typing import Dict
import gym, gym_walk
import torch as pt
import numpy as np
import matplotlib.pyplot as plt

# increase plot resolution
plt.rcParams["figure.dpi"] = 160

# create output directory
output = "output"
makedirs(output, exist_ok=True)

# %%
env = gym.make('SlipperyWalkSeven-v0')
init_state = env.reset()
P = env.env.P
list(P.items())[-2:]


# %%
def value_iteration(P: Dict[int, Dict[int, tuple]], gamma: float=0.99, theta: float=1e-10):
    V = pt.zeros(len(P))
    while True:
        Q = pt.zeros((len(P), len(P[0])))
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if pt.max(pt.abs(V - pt.max(Q, dim=1).values)) < theta:
            break
        V = pt.max(Q, dim=1).values
    pi = lambda s: {s:a for s, a in enumerate(pt.argmax(Q, dim=1))}[s]
    return Q, V, pi


# %%
optimal_Q, optimal_V, optimal_pi = value_iteration(P)
optimal_Q[1:-1]


# %%
def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


# %%
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    Q = pt.zeros((nS, nA))
    select_action = lambda state, Q, epsilon: pt.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, 
                           min_alpha, 
                           alpha_decay_ratio, 
                           n_episodes)
    epsilons = decay_schedule(init_epsilon, 
                              min_epsilon, 
                              epsilon_decay_ratio, 
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * Q[next_state].max() * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)        
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track


# %%
optimal_Q

# %% [markdown]
# ## Components of reinforcement learning

# %% [markdown]
# ## SARSA and Q-learning

# %% [markdown]
# ## Components of deep reinforcement learning

# %% [markdown]
# ## Proximal policy optimization

# %% [markdown]
#

# %%
x = pt.linspace(-0.5, 1.5, 100)
x01 = pt.linspace(0, 1, 50)
mu, std = 0.5, 0.5
gauss = pt.exp(-0.5*((x-mu)/std)**2) / (std * pt.sqrt(pt.tensor(2*np.pi)))
gauss = pt.distributions.Normal(0.5, 0.25)
gauss_pdf = gauss.log_prob(x).exp()
beta = pt.distributions.Beta(2, 2)
beta_pdf = beta.log_prob(x01).exp()

plt.plot(x, gauss_pdf, label=r"Gaussian, $\mu=0.5$, $\sigma = 0.25$")
plt.plot(x01, beta_pdf, label=r"Beta, $\alpha=2$, $\beta = 2$")
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$p(x)$")
plt.text(0.7, 1.5, r"Gauss: $p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$")
plt.text(-0.5, 1.5, r"Beta: $p(x) = \frac{x^{\alpha - 1} (1-x)^{\beta - 1}}{B(\alpha, \beta)}$")
plt.text(-0.5, 1.3, r"$B(\alpha, \beta ) = \int_0^1 x^{\alpha - 1} (1-x)^{\beta - 1} \mathrm{d}x$")
plt.savefig(join(output, "gauss_vs_beta_dist.svg"), bbox_inches="tight")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)

rp = pt.linspace(0, 2, 50)
rn = pt.linspace(-2, 0, 50)
eps = 0.1

ax1.plot(rp, rp, c="k", ls="--", label="original")
ax1.plot(rp, pt.min(rp, rp.clip(1-eps, 1+eps)), c="k", label="clipped")
ax1.axvline(1.0, c="k", ls=":")
ax1.axvline(1.0+eps, c="k", ls=":")
ax1.text(1.15, 0.3, r"$1+\epsilon$", rotation=90)
ax1.set_ylabel(r"$L^{clip}$")
ax1.set_title(r"$A^{GAE} > 0$")
ax1.legend()

ax2.plot(rp, -rp, c="k", ls="--")
ax2.plot(rp, pt.min(-rp, -rp.clip(1-eps, 1+eps)), c="k")
ax2.axvline(1.0, c="k", ls=":")
ax2.axvline(1.0-eps, c="k", ls=":")
ax2.text(0.75, -1.7, r"$1-\epsilon$", rotation=90)
ax2.set_title(r"$A^{GAE} < 0$")

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlabel(r"$r = p^{new}/p^{old}$")
    

plt.savefig(join(output, "ppo_policy_loss_clipping.svg"), bbox_inches="tight")

# %%
