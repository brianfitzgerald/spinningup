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


class ModelA2C(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hid_size: int):
        super(ModelA2C, self).__init__()

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # basic sequential layer
        base_out = self.base(x)
        # compute mean and variance, and value
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class AgentA2C(BaseAgent):
    def __init__(self, net: ModelA2C, device: torch.device):
        self.net = net
        self.device = device

    def __call__(self, states: States, agent_states: AgentStates):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)

        # get the mean and variance of the action
        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        # get the standard deviation
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        # sample from the normal distribution
        actions = np.random.normal(mu, sigma)
        # clip the actions to be within the required range
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.asarray(actions)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


def test_net(
    net: ModelA2C,
    env: gym.Env,
    count: int = 10,
    device: torch.device = torch.device("cpu"),
):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, is_tr, _ = env.step(action)
            rewards += reward  # type: ignore
            steps += 1
            if done or is_tr:
                break
    return rewards / count, steps / count


ENV_IDS = {
    "cheetah": "HalfCheetah-v5",
    "ant": "Ant-v4",
}


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

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                ent_v = -(torch.log(2 * math.pi * var_v) + 1) / 2
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
