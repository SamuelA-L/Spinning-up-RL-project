from spinup import td3_pytorch as td3
import torch
import numpy as np
import gym

env_fn = lambda : gym.make('LunarLander-v2')
# env_fn = lambda : gym.make('Walker2d-v2')


ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='vpg_4x_512')


td3(
    env_fn,
    ac_kwargs=ac_kwargs,
    seed=0, 
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99, 
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000, 
    update_after=1000,
    update_every=50,
    act_noise=0.1,
    target_noise=0.2, 
    noise_clip=0.5,
    policy_delay=2, 
    num_test_episodes=10,
    max_ep_len=1000, 
    logger_kwargs=logger_kwargs,
    save_freq=1
    )