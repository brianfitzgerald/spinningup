"""
Continuous action spaces
"""

import gymnasium as gym
import fire
from loguru import logger


ENV_IDS = {
    "cheetah": "HalfCheetah-v4",
    "ant": "Ant-v4",
}

def main(env_id: str = "cheetah", envs_count: int = 1):

    extra = {}

    env_id = ENV_IDS[env_id]

    envs = [gym.make(env_id, **extra) for _ in range(envs_count)]
    test_env = gym.make(env_id, **extra)
    logger.info(f"Created {envs_count} {env_id} environments.")


if __name__ == "__main__":
    fire.Fire(main)
