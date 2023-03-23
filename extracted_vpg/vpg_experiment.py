from vpg import vpg
import torch
import numpy as np
import gym
import faulthandler

# faulthandler.enable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.Tensor(np.array([1,2,3]))

x.to(device)

env_fn = lambda : gym.make('LunarLander-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='vpg_4x_512')

vpg(
    env_fn,
    ac_kwargs=ac_kwargs,
    seed=0, 
    steps_per_epoch=5000,
    epochs=50,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    logger_kwargs=logger_kwargs,
    save_freq=10
)