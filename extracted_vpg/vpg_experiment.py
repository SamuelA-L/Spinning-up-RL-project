from vpg import vpg
import torch
import numpy as np
import gym
import faulthandler
import pdb


"""
VPG is an on-policy algorithm.

Network architectures.
The on-policy algorithms use networks of size (64, 32) with tanh units for both the policy and the value function.

Batch size.
The on-policy algorithms collected 4000 steps of agent-environment interaction per batch update.
"""


# env_fn = lambda : gym.make('LunarLander-v2')
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)

hidden_sizes = [
    (64, 32),
    (64, 64),
    (128, 64), # *
    (128, 128, 64), # *
    (128, 128, 128)
]

steps_per_epochs = [
    3000,
    4000,
    5000,
    10000
]

for n_steps in steps_per_epochs:
    for size in hidden_sizes:
        exp_name = f"vpg_{env_name}_ac{size}_nsteps{n_steps}-test"
        ac_kwargs = dict(hidden_sizes=size, activation=torch.nn.ReLU)
        logger_kwargs = dict(output_dir='data', exp_name=exp_name)

        vpg(
            env_fn,
            ac_kwargs=ac_kwargs,
            seed=0,
            steps_per_epoch=n_steps,
            logger_kwargs=logger_kwargs,
            save_freq=1
        )