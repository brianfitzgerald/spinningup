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
from typing import List

import fire
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import Tensor

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 32), nn.ReLU(), nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Episode:
    reward: float
    steps: int


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


def iterate_batches(env, net, batch_size):
    """
    perform actions on the environment and collect the results
    until the episode is done, i.e. the environment terminates
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    softmax = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v: Tensor = softmax(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch: List[Episode], percentile: float):
    """
    Find a reward boundary that separates the elite episodes from the non-elite episodes,
    and return the elite episodes and the mean reward.
    An elite episode is one that has a reward greater than the reward boundary,
    so it will should be considered 'successful'.
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


def main():
    env = gymnasium.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(
            "%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
            % (iter_no, loss_v.item(), reward_m, reward_b)
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()

fire.Fire(main)