"""
cross entropy method - on-policy policy-based, model-free method
three main distinctions:
- model free or model based - model free methods learn the policy directly from the environment, while model based methods learn the dynamics of the environment
- policy based or value based - policy based methods learn the policy directly, while value based methods learn the value function and then derive the policy from the value function
- on-policy or off-policy - on-policy methods learn the policy from the current policy, while off-policy methods learn the policy from past data not necessarily generated by the current policy

Neural network provides the policy, which determines the action to take
Randomly sample a probability distribution for an action at each timestep

"""

from dataclasses import dataclass
from typing import List, Tuple

import fire
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import Tensor
import torch.nn.functional as F
from gymnasium import Env, ObservationWrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box, Discrete
from loguru import logger
from typing import Literal


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


EnvironmentChoice = Literal["cartpole", "frozenlake"]


@dataclass
class Episode:
    reward: float
    steps: int


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


def iterate_batches(env: Env, net: nn.Module, batch_size: int):
    """
    perform actions on the environment and collect the results
    until the episode is done, i.e. the environment terminates
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    while True:
        obs_v = torch.FloatTensor(obs)
        act_probs_v: Tensor = F.softmax(net(obs_v), dim=0)
        act_probs = act_probs_v.data.numpy()
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # logger.debug(f"obs: {obs}, action: {action}, reward: {reward} term/trunc: {terminated}/{truncated}")
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if terminated or truncated:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
            next_obs, _ = env.reset()
        obs = next_obs


def filter_batch(
    batch: List[Episode],
    percentile: float,
    gamma: float = 0.9,
) -> Tuple[List[Episode], List[np.ndarray], List[int], float]:
    """
    Find a reward boundary that separates the elite episodes from the non-elite episodes,
    and return the elite episodes and the mean reward.
    An elite episode is one that has a reward greater than the reward boundary,
    so it will be considered 'successful'.

    Optionally discount the reward, to reduce it over each step, which penalizes longer episodes.
    """
    rewards = list(map(lambda s: s.reward * gamma ** len(s.steps), batch))
    reward_bound = np.percentile(rewards, percentile)

    train_obs: List[np.ndarray] = []
    train_act: List[int] = []
    elite_batch: List[Episode] = []

    for example, reward in zip(batch, rewards):
        if reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


class DiscreteOneHotWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, Discrete)
        shape = (env.observation_space.n,)
        # Box is a set of intervals in n-dimensional space
        self.observation_space: Box = Box(0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        # set the tile we are on to 1.0, and others to 0.0
        res[observation] = 1.0
        return res


def main(environment: EnvironmentChoice = "CartPole"):
    env_name = "CartPole-v1" if environment == "cartpole" else "FrozenLake-v1"

    env = gymnasium.make(env_name, render_mode="rgb_array")
    if environment == "frozenlake":
        env = DiscreteOneHotWrapper(env)

    env = RecordVideo(env, video_folder=f"videos/chapter_4/{env_name}")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if environment == "cartpole":
        hidden_size = 16
        batch_size = 16
        percentile = 70
        lr = 0.01
    elif environment == "frozenlake":
        hidden_size = 128
        batch_size = 16
        percentile = 30
        lr = 0.001

    net = Net(obs_size, hidden_size, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    writer = SummaryWriter(comment=f"-{environment}")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, batch_size)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        gamma = 1 if environment == "cartpole" else 0.9
        full_batch, obs, acts, reward_bound = filter_batch(batch, percentile, gamma)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        # the value scores for each action
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        logger.info(
            "%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
            % (iter_no, loss_v.item(), reward_mean, reward_bound)
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean >= 0.8:
            print("Solved!")
            break
    writer.close()


fire.Fire(main)
