"""
DDPG - Deep Deterministic Policy Gradient
Off policy actor-crtic method
Deterministic policy gradient, which means it directly provies the action to take from the state
Means we can apply the chain rule to the Q-value, and maximize the Q-value with respect to the policy parameters

Actor gives the action to take for each state - N values, one for each action
Critic gives the value of the state-action pair - aka Q-value
Subsitute the Q-value into the Bellman equation, and get the loss function - Q(s, u(s)) where u(s) is the action from the actor
Stochastic policy gradient is the same as the deterministic policy gradient, but with the addition of the entropy term
In A2C, critic is used as the baseline, but in DDPG, the critic is used to train the actor
Since the policy is deterministic, we can differentiate the Q-value with respect to the action, and use it to train the actor with SGD
Use the Bellman equation to train the critic, and SGD for the actor
The critic is updated similar to A2C, and actor is updated using the Q-value gradient, which maximizes the critic's output
The method is off-policy, so we can use the replay buffer to store the experiences, and sample from it to train the network

Since the policy is deterministic, we can't use the entropy term, so we use the Ornstein-Uhlenbeck process to add noise to the action
Models the velocity of the Brownian motion, and adds it to the action, i.e. normal noise
Critic has two paths - one for observation, and one for the action
Actor is just observations -> actions
"""

import math
import os
import time
import sys

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import RewardTracker, ensure_directory, get_device
from loguru import logger
from ptan import (
    AgentStates,
    BaseAgent,
    ExperienceSourceFirstLast,
    States,
    TBMeanTracker,
    float32_preprocessor,
)
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium.wrappers import RecordVideo


ENV_IDS = {
    "cheetah": "HalfCheetah-v5",
    "ant": "Ant-v4",
}


class DDPGActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300), nn.ReLU(), nn.Linear(300, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        # a is the action, and x is the observation
        obs = self.obs_net(x)
        # Concatenate the observation and action, and pass it through the output network
        # Return the Q-value, which is the value of the state-action pair
        return self.out_net(torch.cat([obs, a], dim=1))


GAMMA = 0.98
REWARD_STEPS = 2
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 100


def calc_logprob(mu_v: torch.Tensor, var_v: torch.Tensor, actions_v: torch.Tensor):
    # differential entropy of the normal distribution
    p1 = -((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    # sample from the normal distribution
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def main(env_id: str = "cheetah", envs_count: int = 1):

    device_name = get_device()
    device = torch.device(device_name)

    env_id = ENV_IDS[env_id]

    ensure_directory("videos", clear=True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join("videos", "a2c_" + env_id))
    test_env = gym.make(env_id)
    logger.info(f"Created {envs_count} {env_id} environments.")

    obs_size, act_size = (
        env.observation_space.shape[0],  # type: ignore
        env.action_space.shape[0],  # type: ignore
    )

    net = ModelA2C(
        obs_size=obs_size,
        act_size=act_size,
        hid_size=64,
    ).to(device)

    writer = SummaryWriter(comment=f"-a2c_{env_id}")
    agent = AgentA2C(net, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    save_path = os.path.join("saves", "a2c_" + env_id)

    batch = []
    best_reward = None
    with RewardTracker(writer, sys.maxsize) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print(
                        "Test done is %.2f sec, reward %.3f, steps %d"
                        % (time.time() - ts, rewards, steps)
                    )
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(
                                "Best reward updated: %.3f -> %.3f"
                                % (best_reward, rewards)
                            )
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            ensure_directory(save_path)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = unpack_batch_a2c(
                    batch, net, device=device_name, last_val_gamma=GAMMA**REWARD_STEPS
                )
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                # Value loss
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                # Advantage loss
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                # Policy loss
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                ent_v = -(torch.log(2 * math.pi * var_v) + 1) / 2
                # Entropy loss
                entropy_loss_v = ENTROPY_BETA * ent_v.mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)


if __name__ == "__main__":
    fire.Fire(main)
