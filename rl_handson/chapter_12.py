"""
REINFORCE has lots of problems that lead to high variance of the gradient.
Variance is the expectation of the squared deviation of a random variable from its mean.
In order to reduce the variance, we can subtract a baseline from the return.

A2C - advantage actor-critic
Idea is that we can represent the reward as the valute of a state plus the advantage of taking an action in that state.
However, we can't use V(s) as a baseline, since we don't know the value of the state that we need to subtract from the discounted total reward.
Instead, use another neural network to estimate the value of the state, and use this as the baseline.
To train it, carry out the Bellman step and then minimize the MSE between the predicted value and the actual return.

The 'critic' part of the A2C algorithm is the value network, which estimates the value of the state. Some additions:
- Entropy bonus to improve exploration
- Gradient accumulation to improve stability, merging the policy and value networks into a single network
- Training with several environments in parallel to improve sample efficiency

"""

import typing as tt

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from loguru import logger
from models import AtariA2C, SimpleLinear
from ptan import (
    ExperienceFirstLast,
    ExperienceSourceFirstLast,
    PolicyAgent,
    float32_preprocessor,
)
from torch.utils.tensorboard.writer import SummaryWriter

from rl_handson.lib import get_device, wrap_dqn

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


def unpack_batch(
    batch: tt.List[ExperienceFirstLast],
    net: AtariA2C,
    device: torch.device,
    gamma: float,
    reward_steps: int,
):
    """
    Convert batch into training tensors
    :param batch: batch to process
    :param net: network to useß
    :param gamma: gamma value
    :param reward_steps: steps of reward
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))

    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_t = net(last_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma**reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(device)

    return states_t, actions_t, ref_vals_t


def main(use_async: bool = True):

    device = torch.device(get_device())

    env_factories = [
        lambda: wrap_dqn(gym.make("PongNoFrameskip-v4")) for _ in range(NUM_ENVS)
    ]
    if use_async:
        env = gym.vector.AsyncVectorEnv(env_factories)
    else:
        env = gym.vector.SyncVectorEnv(env_factories)
    writer = SummaryWriter(comment="-cartpole-reinforce")
    env = RecordVideo(env, video_folder=f"videos/chapter_11/cartpole")

    net = SimpleLinear(env.observation_space.shape[0], env.action_space.n)

    agent = PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)
    # Experience source that returns the first and last states only
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []
    writer.close()


if __name__ == "__main__":
    fire.Fire(main)
