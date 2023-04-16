from vpg import vpg
import torch
import numpy as np
import gym
import faulthandler
import pdb

"""
from grid search
    data/vpg_HalfCheetah-v2_ac\(128,\ 64\)_nsteps5000/
"""
size = (128, 64)
n_steps = 5000
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)

for i in range(1) :

    exp_name = f"best_vpg_{env_name}_ac{size}_nsteps{n_steps}_seed{i}"
    ac_kwargs = dict(hidden_sizes=size, activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='data', exp_name=exp_name)

    # profiler = torch.autograd.profile(
        # schedule=torch.cuda.profiler.schedule(),
        # on_trace_ready=torch.cuda.profiler.tensorboard_trace_handler(f"data/{exp_name}/{exp_name}.prof"),
        # record_shapes=True,
        # with_stack=True
    # )

    with torch.cuda.profile() as prof:
        vpg(
            env_fn,
            ac_kwargs=ac_kwargs,
            seed=i,
            steps_per_epoch=n_steps,
            logger_kwargs=logger_kwargs,
            save_freq=1
        )
