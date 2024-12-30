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
from typing import Literal

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
    ExperienceReplayBuffer,
    ExperienceSourceFirstLast,
    States,
    TBMeanTracker,
    TargetNet,
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


Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

def distr_projection(
        next_distr: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float
):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS),
                          dtype=np.float32)
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    for atom in range(N_ATOMS):
        v = rewards + (Vmin + atom * delta_z) * gamma
        tz_j = np.minimum(Vmax, np.maximum(Vmin, v))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += \
            next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += \
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += \
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(
            Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = \
                (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = \
                (b_j - l)[ne_mask]
    return proj_distr



class D4PGCritic(nn.Module):
    def __init__(
        self, obs_size: int, act_size: int, n_atoms: int, v_min: float, v_max: float
    ):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300), nn.ReLU(), nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr: torch.Tensor):
        # Weighted sum of the supports
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)


class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(
        self,
        net: DDPGActor,
        device: torch.device = torch.device("cpu"),
        ou_enabled: bool = True,
        ou_mu: float = 0.0,
        ou_teta: float = 0.15,
        ou_sigma: float = 0.2,
        ou_epsilon: float = 1.0,
    ):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states: States, agent_states: AgentStates):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                # Add noise to the agent state
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states



def test_net(
    net: DDPGActor,
    env: gym.Env,
    count: int = 10,
    device: torch.device = torch.device("cpu"),
):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs, _ = env.reset()
        while True:
            obs_v = float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, is_tr, _ = env.step(action)
            rewards += reward  # type: ignore
            steps += 1
            if done or is_tr:
                break
    return rewards / count, steps / count


def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = float32_preprocessor(states).to(device)
    actions_v = float32_preprocessor(actions).to(device)
    rewards_v = float32_preprocessor(rewards).to(device)
    last_states_v = float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


AlgorithmChoice = Literal["ddpg", "d4pg"]


def main(env_id: str = "cheetah", envs_count: int = 1, algo: AlgorithmChoice = "d4pg"):

    device_name = get_device()
    device = torch.device(device_name)

    env_id = ENV_IDS[env_id]

    if algo == "ddpg":
        GAMMA = 0.99
        BATCH_SIZE = 64
        LEARNING_RATE = 1e-4
        REPLAY_SIZE = 100000
        REPLAY_INITIAL = 10000
    else:
        GAMMA = 0.99
        BATCH_SIZE = 64
        LEARNING_RATE = 1e-4
        REPLAY_SIZE = 100000
        REPLAY_INITIAL = 10000
        REWARD_STEPS = 5

    TEST_ITERS = 100

    ensure_directory("videos", clear=True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, os.path.join("videos", "ddpg_" + env_id))
    test_env = gym.make(env_id)
    logger.info(f"Created {envs_count} {env_id} environments.")

    obs_size, act_size = (
        env.observation_space.shape[0],  # type: ignore
        env.action_space.shape[0],  # type: ignore
    )
    act_net = DDPGActor(obs_size, act_size).to(device)
    if algo == "d4pg":
        crt_net = D4PGCritic(obs_size, act_size, 51, -500, 500).to(device)
    else:
        crt_net = DDPGCritic(obs_size, act_size).to(device)
    writer = SummaryWriter(comment=f"-ddpg_{env_id}")
    agent = AgentDDPG(act_net, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)

    tgt_act_net = TargetNet(act_net)
    tgt_crt_net = TargetNet(crt_net)

    save_path = os.path.join("saves", "ddpg_" + env_id)

    buffer = ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with RewardTracker(writer, sys.maxsize) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = (
                    unpack_batch_ddqn(batch, device_name)
                )

                # TODO add d4pg loss fn
                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                # soft sync target networks
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print(
                        "Test done in %.2f sec, reward %.3f, steps %d"
                        % (time.time() - ts, rewards, steps)
                    )
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(
                                "Best reward updated: %.3f -> %.3f"
                                % (best_reward, rewards)
                            )
                            ensure_directory(save_path)
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards


if __name__ == "__main__":
    fire.Fire(main)
