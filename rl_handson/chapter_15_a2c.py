"""
Continuous action spaces
Idea is to sample from a continuous action space for each parameter, instead of sampling probability of each action.
Can apply A2C to continuous action spaces by using a Gaussian distribution to sample actions.
Represent the policy as a Gaussian distribution with mean and standard deviation.
So the network will output two values for each action, mean and standard deviation.
https://gymnasium.farama.org/environments/mujoco/half_cheetah/
"""

import math
import os

import fire
import gymnasium as gym
import torch
from lib import MUJOCO_ENV_IDS, ensure_directory, get_device
from loguru import logger
from gymnasium.wrappers import RecordVideo


def calc_logprob(mu_v: torch.Tensor, var_v: torch.Tensor, actions_v: torch.Tensor):
    # differential entropy of the normal distribution
    p1 = -((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    # sample from the normal distribution
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def main(env_id: str = "cheetah", envs_count: int = 1):

    device_name = get_device()
    device = torch.device(device_name)

    env_id = MUJOCO_ENV_IDS[env_id]

    ensure_directory("videos", clear=True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join("videos", "a2c_" + env_id))
    test_env = gym.make(env_id)
    logger.info(f"Created {envs_count} {env_id} environments.")

    obs_size, act_size = (
        env.observation_space.shape[0],  # type: ignore
        env.action_space.shape[0],  # type: ignore
    )


if __name__ == "__main__":
    fire.Fire(main)
