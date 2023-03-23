from vpg import vpg
import torch
import numpy as np
import gym
import faulthandler
import pdb



# env_fn = lambda : gym.make('LunarLander-v2')
env_fn = lambda : gym.make('Walker2d-v2')


ac_kwargs = dict(hidden_sizes=[256,256,256])

logger_kwargs = dict(output_dir='data', exp_name='profile_vpg')

vpg(
    env_fn,
    ac_kwargs=ac_kwargs,
    seed=0, 
    steps_per_epoch=5000,
    epochs=5,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    logger_kwargs=logger_kwargs,
    save_freq=10
)