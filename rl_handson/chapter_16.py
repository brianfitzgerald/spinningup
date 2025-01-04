"""
Establish baseline with A2C
Then implement PPO

PPO - instead of using the gradient of the log probability of the action, we use the ratio of the new policy to the old policy, scaled by the advantage

In math form, A2C gradient is E[log(pi(a|s)) * A(s, a)]
PPO gradient is E[pi_new(a|s) / pi_old(a|s) * A(s, a)]
Similar to cross entropy loss, this represents importance sampling
However, if we just blindly update teh value with this, it can lead to large updates
To prevent this, we clip the ratio to be within a certain range
Specifically, take the minimum of the clipped ratio and the unclipped ratio
i.e. min(ratio, clip(ratio, 1 - epsilon, 1 + epsilon))
so if the unclipped value is lower, use that; otherwise, use the clipped value
PPO also uses a more general advantage estimate, called GAE (Generalized Advantage Estimation)
the equation is A(s, a) = delta_t + gamma * lambda * delta_t+1 + gamma^2 * lambda^2 * delta_t+2 + ...
i.e. the advantage is the sum of the discounted rewards, but with a lambda factor that decreases the importance of future rewards
Skipping TRPO since it's complicated and not really used

SAC - actor-critic with entropy regularization bonus
Give the agent a bonus for exploration
Policy is argmax of Q(s, a) + alpha * H(pi(s))
Alpha is a hyperparameter that controls the importance of the entropy bonus
Also uses clipped double-Q trick - use two Q networks, and take the minimum of the two
This helps with dealing with Q-value over-estimatiomn during training
Trained with MSE objective by doing Bellman approximation using the target value network

Model actor - policy variance is not parameterized by the state - log_std is just at ensor, not a network - which in classical SAC isn't the case
Critic is the same

If an environment is fast and observeations are easy to obtain, on policy methods are a better choice
But if observations are hard to obtain, off-policy methods will do better
"""

import math
import os
import time
from typing import List, Literal, Optional, Sequence

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.distributions as distr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo, TimeLimit
from lib import MUJOCO_ENV_IDS, RewardTracker, ensure_directory, get_device
from loguru import logger
from models import ModelSACTwinQ
from ptan import (
    AgentStates,
    BaseAgent,
    Experience,
    ExperienceFirstLast,
    ExperienceReplayBuffer,
    ExperienceSourceFirstLast,
    States,
    TargetNet,
    TBMeanTracker,
    float32_preprocessor,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class ModelActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 64):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int = 64):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
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
    count: int = 1,
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
    batch: Sequence[ExperienceFirstLast],
    net: nn.Module,
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


def calc_adv_ref(
    trajectory: List[Experience],
    net_crt: nn.Module,
    states_v: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    device: str,
):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, exp in zip(
        reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])
    ):
        if exp.done_trunc:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(np.asarray(list(reversed(result_adv))))
    ref_v = torch.FloatTensor(np.asarray(list(reversed(result_ref))))
    return adv_v.to(device), ref_v.to(device)


def unpack_batch_ppo(
    trajectory: List[ExperienceFirstLast],
    device: str,
):
    traj_states = [t.state for t in trajectory]
    traj_actions = [t.action for t in trajectory]
    traj_states_v = torch.FloatTensor(np.asarray(traj_states))
    traj_states_v = traj_states_v.to(device)
    traj_actions_v = torch.FloatTensor(np.asarray(traj_actions))
    traj_actions_v = traj_actions_v.to(device)
    return traj_states_v, traj_actions_v


@torch.no_grad()
def unpack_batch_sac(
    batch: Sequence[ExperienceFirstLast],
    val_net: nn.Module,
    twinq_net: nn.Module,
    policy_net: ModelActor,
    gamma: float,
    ent_alpha: float,
    device: torch.device,
):
    """
    Unpack Soft Actor-Critic batch
    """
    states_v, actions_v, ref_q_v = unpack_batch_a2c(batch, val_net, gamma, device)

    # references for the critic network
    mu_v = policy_net(states_v)
    act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    # sample actions from the distribution of the policy net
    acts_v = act_dist.sample()
    q1_v, q2_v = twinq_net(states_v, acts_v)
    # element-wise minimum of the two q-networks, then subtract the entropy bonus
    ref_vals_v = torch.min(q1_v, q2_v).squeeze() - ent_alpha * act_dist.log_prob(
        acts_v
    ).sum(dim=1)
    return states_v, actions_v, ref_vals_v, ref_q_v


def calc_logprob(mu_v: torch.Tensor, logstd_v: torch.Tensor, actions_v: torch.Tensor):
    # calculate the log of the probability of the action
    # This is log(N(x|mu, sigma))
    p1 = -((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    # log of the square root of 2 * pi * sigma^2
    p2 = -torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


AlgorithmChoice = Literal["a2c", "ppo", "sac"]


def main(
    env_id: str = "ant",
    save_path: str = "saves",
    checkpoint: Optional[str] = None,
    envs_count: int = 50,
    algorithm: AlgorithmChoice = "sac",
):
    env_id = MUJOCO_ENV_IDS[env_id]
    envs = [gym.make(env_id) for _ in range(envs_count)]
    test_env = gym.make(env_id, render_mode="rgb_array")
    ensure_directory(save_path, True)
    video_path = os.path.join("videos", f"{algorithm}-{env_id}")
    ensure_directory(video_path, True)
    test_env = RecordVideo(test_env, video_path)
    test_env = TimeLimit(test_env, max_episode_steps=1000)

    logger.info(f"Training {algorithm} on env {env_id}")

    GAMMA = 0.99
    REWARD_STEPS = 5
    BATCH_SIZE = 32
    TEST_ITERS = 100000
    MEAN_TRACKER_BATCH_SIZE = 100

    if algorithm == "a2c":
        LEARNING_RATE_ACTOR = 1e-5
        LEARNING_RATE_CRITIC = 1e-3
    elif algorithm == "ppo":
        LEARNING_RATE_ACTOR = 1e-5
        LEARNING_RATE_CRITIC = 1e-4
    elif algorithm == "sac":
        GAMMA = 0.99
        BATCH_SIZE = 64
        LEARNING_RATE_ACTOR = 1e-4
        LEARNING_RATE_CRITIC = 1e-4
        TEST_ITERS = 10000
        MEAN_TRACKER_BATCH_SIZE = 10

    # only used for SAC
    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000
    SAC_ENTROPY_ALPHA = 0.1
    LR_VALS = 1e-4

    # only used for A2C
    ENTROPY_BETA = 1e-3

    # only used for PPO
    PPO_EPS = 0.2
    PPO_EPOCHS = 10
    PPO_BATCH_SIZE = 64
    GAE_LAMBDA = 0.95

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

    writer = SummaryWriter(comment=f"-{algorithm}-{env_id}")
    agent = AgentA2C(net_act, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)

    if checkpoint:
        net_act.load_state_dict(torch.load(checkpoint))
        logger.info("Loaded from checkpoint %s" % checkpoint)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = []
    best_reward = None

    frame_idx = 0
    with RewardTracker(writer) as tracker:
        with TBMeanTracker(writer, batch_size=MEAN_TRACKER_BATCH_SIZE) as tb_tracker:
            if algorithm == "sac":
                twinq_net = ModelSACTwinQ(obs_size, act_size, 64).to(device)
                twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)
                tgt_crt_net = TargetNet(net_crt)
                buffer = ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
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

                    batch: List[ExperienceFirstLast] = buffer.sample(BATCH_SIZE)  # type: ignore
                    states_v, actions_v, ref_vals_v, ref_q_v = unpack_batch_sac(
                        batch,
                        tgt_crt_net.target_model,
                        twinq_net,
                        net_act,
                        GAMMA,
                        SAC_ENTROPY_ALPHA,
                        device,
                    )

                    tb_tracker.track("ref_v", ref_vals_v.mean(), frame_idx)
                    tb_tracker.track("ref_q", ref_q_v.mean(), frame_idx)

                    # train TwinQ
                    twinq_opt.zero_grad()
                    q1_v, q2_v = twinq_net(states_v, actions_v)
                    q1_loss_v = F.mse_loss(q1_v.squeeze(), ref_q_v.detach())
                    q2_loss_v = F.mse_loss(q2_v.squeeze(), ref_q_v.detach())
                    # Sum the losses from the two Q networks
                    q_loss_v = q1_loss_v + q2_loss_v
                    q_loss_v.backward()
                    twinq_opt.step()
                    tb_tracker.track("loss_q1", q1_loss_v, frame_idx)
                    tb_tracker.track("loss_q2", q2_loss_v, frame_idx)

                    # Critic
                    opt_crt.zero_grad()
                    val_v = net_crt(states_v)
                    # Just MSE loss between the value and the reference value
                    v_loss_v = F.mse_loss(val_v.squeeze(), ref_vals_v.detach())
                    v_loss_v.backward()
                    opt_crt.step()
                    tb_tracker.track("loss_v", v_loss_v, frame_idx)

                    # Actor
                    opt_act.zero_grad()
                    acts_v = net_act(states_v)
                    q_out_v, _ = twinq_net(states_v, acts_v)
                    # loss is the negative of the Q value
                    act_loss = -q_out_v.mean()
                    act_loss.backward()
                    opt_act.step()
                    tb_tracker.track("loss_act", act_loss, frame_idx)

                    tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if frame_idx % TEST_ITERS == 0:
                        ts = time.time()
                        rewards, steps = test_net(net_act, test_env, device=device)
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
                                name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                                fname = os.path.join(save_path, name)
                                torch.save(net_act.state_dict(), fname)
                            best_reward = rewards

            else:
                for step_idx, exp in enumerate(exp_source):
                    # same between A2C and PPO
                    rewards_steps = exp_source.pop_rewards_steps()
                    if rewards_steps:
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                        tracker.reward(np.mean(rewards), step_idx)

                    # same between A2C and PPO
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
                                name = f"{algorithm}-best_{rewards:.0f}.dat"
                                fname = os.path.join(save_path, name)
                                torch.save(net_act.state_dict(), fname)
                            best_reward = rewards

                    trajectory.append(exp)
                    if len(trajectory) < BATCH_SIZE:
                        continue

                    if algorithm == "a2c":
                        states_v, actions_v, vals_ref_v = unpack_batch_a2c(
                            trajectory,
                            net_crt,
                            last_val_gamma=GAMMA**REWARD_STEPS,
                            device=device,
                        )
                        trajectory.clear()

                        opt_crt.zero_grad()
                        value_v = net_crt(states_v)
                        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                        loss_value_v.backward()
                        opt_crt.step()

                        opt_act.zero_grad()
                        mu_v = net_act(states_v)
                        # get advantage from critic
                        adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                        # calculate log of mu/std, and multiply by advantage
                        log_prob_v = adv_v * calc_logprob(
                            mu_v, net_act.logstd, actions_v
                        )
                        loss_policy_v = -log_prob_v.mean()
                        # Entropy loss
                        entropy_loss_v = (
                            ENTROPY_BETA
                            * (
                                -(
                                    torch.log(2 * math.pi * torch.exp(net_act.logstd))
                                    + 1
                                )
                                / 2
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

                    elif algorithm == "ppo":
                        traj_states_v, traj_actions_v = unpack_batch_ppo(
                            trajectory, device_str
                        )
                        traj_adv_v, traj_ref_v = calc_adv_ref(
                            trajectory,
                            net_crt,
                            traj_states_v,
                            GAMMA,
                            GAE_LAMBDA,
                            device_str,
                        )
                        mu_v = net_act(traj_states_v)
                        old_logprob_v = calc_logprob(
                            mu_v, net_act.logstd, traj_actions_v
                        )

                        # normalize advantages
                        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
                        traj_adv_v /= torch.std(traj_adv_v)

                        # drop last entry from the trajectory, an our adv and ref value calculated without it
                        trajectory = trajectory[:-1]
                        old_logprob_v = old_logprob_v[:-1].detach()

                        sum_loss_value = 0.0
                        sum_loss_policy = 0.0
                        count_steps = 0

                        for _ in range(PPO_EPOCHS):
                            for batch_ofs in range(0, len(trajectory), PPO_BATCH_SIZE):
                                batch_l = batch_ofs + PPO_BATCH_SIZE
                                states_v = traj_states_v[batch_ofs:batch_l]
                                actions_v = traj_actions_v[batch_ofs:batch_l]
                                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                                batch_adv_v = batch_adv_v.unsqueeze(-1)
                                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                                # critic training
                                opt_crt.zero_grad()
                                value_v = net_crt(states_v)
                                # simply MSE loss between value_v and batch_ref_v
                                loss_value_v = F.mse_loss(
                                    value_v.squeeze(-1), batch_ref_v
                                )
                                loss_value_v.backward()
                                opt_crt.step()

                                # actor training
                                opt_act.zero_grad()
                                mu_v = net_act(states_v)
                                logprob_pi_v = calc_logprob(
                                    mu_v, net_act.logstd, actions_v
                                )
                                # get the ratio of the new policy to the old policy
                                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                                surr_obj_v = batch_adv_v * ratio_v

                                # clipped ratio of the new policy to the old policy
                                c_ratio_v = torch.clamp(
                                    ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS
                                )
                                # clipped surrogate objective - advantage * clipped ratio
                                clipped_surr_v = batch_adv_v * c_ratio_v
                                loss_policy_v = -torch.min(
                                    surr_obj_v, clipped_surr_v
                                ).mean()
                                loss_policy_v.backward()
                                opt_act.step()

                                sum_loss_value += loss_value_v.item()
                                sum_loss_policy += loss_policy_v.item()
                                count_steps += 1

                        trajectory.clear()
                        writer.add_scalar(
                            "advantage", traj_adv_v.mean().item(), step_idx
                        )
                        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
                        writer.add_scalar(
                            "loss_policy", sum_loss_policy / count_steps, step_idx
                        )
                        writer.add_scalar(
                            "loss_value", sum_loss_value / count_steps, step_idx
                        )
    for e in envs:
        e.close()


if __name__ == "__main__":
    fire.Fire(main)
