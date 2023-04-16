from td3 import td3
import torch
import numpy as np
import gym


"""
TD3 is an off-policy algorithm.

Network architectures.
The off-policy algorithms use networks of size (256, 256) with relu units.

Batch size.
The off-policy algorithms used minibatches of size 100 at each gradient descent step.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)

hidden_sizes = [
    (256,256),
    (256,256,128),
    (256,256,256),
    (256,256,256,256),

]

batch_sizes = [
    100,
    500,
    1000
]

for batch_size in batch_sizes :
    for size in hidden_sizes :
        exp_name = f"td3_{env_name}_ac{size}_batch{batch_size}-test"
        ac_kwargs = dict(hidden_sizes=size, activation=torch.nn.ReLU)
        logger_kwargs = dict(output_dir='data', exp_name=exp_name)


        td3(
            env_fn,
            ac_kwargs=ac_kwargs,
            seed=0,
            batch_size=batch_size,
            logger_kwargs=logger_kwargs,
            save_freq=1
            )