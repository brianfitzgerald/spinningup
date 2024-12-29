"""
Continuous action spaces
Idea is to sample from a continuous action space for each parameter, instead of sampling probability of each action.
Can apply A2C to continuous action spaces by using a Gaussian distribution to sample actions.
Represent the policy as a Gaussian distribution with mean and standard deviation.
So the network will output two values for each action, mean and standard deviation.
"""

import gymnasium as gym
import fire
from loguru import logger
import torch.nn as nn
import torch

from lib import get_device


class ModelA2Continuous(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hid_size: int):
        super(ModelA2Continuous, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(hid_size, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(hid_size, 1)

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


ENV_IDS = {
    "cheetah": "HalfCheetah-v4",
    "ant": "Ant-v4",
}


GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16

TEST_ITERS = 100000


def main(env_id: str = "cheetah", envs_count: int = 1):

    extra = {}
    device = get_device()

    env_id = ENV_IDS[env_id]

    envs = [gym.make(env_id, **extra) for _ in range(envs_count)]
    test_env = gym.make(env_id, **extra)
    logger.info(f"Created {envs_count} {env_id} environments.")

    obs_size, act_size = (
        envs[0].observation_space.shape[0],
        envs[0].action_space.shape[0],
    )

    model = ModelA2Continuous(
        obs_size=obs_size,
        act_size=act_size,
        hid_size=64,
    ).to(device)


if __name__ == "__main__":
    fire.Fire(main)
