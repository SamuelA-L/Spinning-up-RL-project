from td3 import td3
import torch
import numpy as np
import gym

"""
from grid search
 data/td3_HalfCheetah-v2_ac\(256,\ 256,\ 256\)_batch100
"""
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)
size = (256, 256, 256)
batch_size = 100

for i in range(5):
    exp_name = f"best_td3_{env_name}_ac{size}_batch{batch_size}_seed{i}"
    ac_kwargs = dict(hidden_sizes=size, activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='data', exp_name=exp_name)

    profiler = torch.profiler.profile(
        schedule=torch.cuda.profiler.schedule(),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"data/{exp_name}/{exp_name}.prof"),
        record_shapes=True,
        with_stack=True)

    profiler.start()
    td3(
        env_fn,
        ac_kwargs=ac_kwargs,
        seed=i,
        batch_size=batch_size,
        logger_kwargs=logger_kwargs,
        save_freq=1
    )
    profiler.stop()