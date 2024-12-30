#!/usr/bin/env python3
import math
import os
import time

import fire
import gymnasium as gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
from lib import MUJOCO_ENV_IDS, RewardTracker, ensure_directory, get_device
from typing import List, Optional
from gymnasium.wrappers import RecordVideo, TimeLimit
from tqdm import tqdm
from loguru import logger

from ptan import (
    AgentStates,
    BaseAgent,
    ExperienceFirstLast,
    States,
    float32_preprocessor,
)

GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16
HID_SIZE = 64

TEST_ITERS = 100000


class ModelActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size: int):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x)


class AgentA2C(BaseAgent):
    def __init__(self, net, device: torch.device):
        self.net = net
        self.device = device

    def __call__(self, states: States, agent_states: AgentStates):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, -1, 1)
        return actions, agent_states


def test_net(
    net: ModelActor,
    env: gym.Env,
    count: int = 2,
    device: torch.device = torch.device("cpu"),
):
    rewards = 0.0
    steps = 0
    iterator = tqdm(range(count))
    for _ in iterator:
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
            iterator.set_postfix({"rewards": rewards, "steps": steps})
            if done or is_tr:
                break
    return rewards / count, steps / count


def unpack_batch_a2c(
    batch: List[ExperienceFirstLast],
    net: ModelCritic,
    last_val_gamma: float,
    device: torch.device,
):
    """
    Convert batch into training tensors
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
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


def calc_logprob(mu_v: torch.Tensor, logstd_v: torch.Tensor, actions_v: torch.Tensor):
    p1 = -((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


def main(
    env_id: str = "cheetah",
    name: str = "a2c",
    save_path: str = "saves",
    checkpoint: Optional[str] = None,
):
    env_id = MUJOCO_ENV_IDS[env_id]
    envs = [gym.make(env_id) for _ in range(ENVS_COUNT)]
    test_env = gym.make(env_id, render_mode="rgb_array")
    ensure_directory(save_path, True)
    video_path = os.path.join("videos", f"{name}-{env_id}")
    ensure_directory(video_path, True)
    test_env = RecordVideo(test_env, video_path)
    test_env = TimeLimit(test_env, max_episode_steps=1000)

    device_str = get_device()
    device = torch.device(device_str)

    obs_size, act_size = (
        envs[0].observation_space.shape[0],  # type: ignore
        envs[0].action_space.shape[0],  # type: ignore
    )

    net_act = ModelActor(obs_size, act_size).to(device)
    net_crt = ModelCritic(obs_size).to(device)
    logger.info(net_act)
    logger.info(net_crt)

    writer = SummaryWriter(comment="-a2c_" + name)
    agent = AgentA2C(net_act, device=device)
    exp_source = ptan.ExperienceSourceFirstLast(
        envs, agent, GAMMA, steps_count=REWARD_STEPS
    )

    if checkpoint:
        net_act.load_state_dict(torch.load(checkpoint))
        logger.info("Loaded from checkpoint %s" % checkpoint)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with RewardTracker(writer) as tracker:
        with ptan.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    print("Test started")
                    rewards, steps = test_net(net_act, test_env, device=device)
                    print(
                        "Test done in %.2f sec, reward %.3f, steps %d"
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
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = unpack_batch_a2c(
                    batch, net_crt, last_val_gamma=GAMMA**REWARD_STEPS, device=device
                )
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, net_act.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = (
                    ENTROPY_BETA
                    * (
                        -(torch.log(2 * math.pi * torch.exp(net_act.logstd)) + 1) / 2
                    ).mean()
                )
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)


if __name__ == "__main__":
    fire.Fire(main)
