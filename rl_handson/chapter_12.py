"""
REINFORCE has lots of problems that lead to high variance of the gradient.
Variance is the expectation of the squared deviation of a random variable from its mean.
In order to reduce the variance, we can subtract a baseline from the return.

A2C - advantage actor-critic
Idea is that we can represent the reward as the valute of a state plus the advantage of taking an action in that state.
However, we can't use V(s) as a baseline, since we don't know the value of the state that we need to subtract from the discounted total reward.
Instead, use another neural network to estimate the value of the state, and use this as the baseline.
To train it, carry out the Bellman step and then minimize the MSE between the predicted value and the actual return.

The 'critic' part of the A2C algorithm is the value network, which estimates the value of the state. Some additions:
- Entropy bonus to improve exploration
- Gradient accumulation to improve stability, merging the policy and value networks into a single network
- Training with several environments in parallel to improve sample efficiency

"""

import typing as tt
import ale_py

from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import fire
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from lib import RewardTracker, ensure_directory, get_device, wrap_dqn
from models import AtariA2C
from ptan import (
    ExperienceFirstLast,
    PolicyAgent,
    TBMeanTracker,
    VectorExperienceSourceFirstLast,
    float32_preprocessor,
)
from torch.utils.tensorboard.writer import SummaryWriter
from loguru import logger

gym.register_envs(ale_py)

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


def unpack_batch(
    batch: tt.List[ExperienceFirstLast],
    net: AtariA2C,
    device: torch.device,
    gamma: float,
    reward_steps: int,
):
    """
    Convert batch into training tensors
    :param batch: batch to process
    :param net: network to useß
    :param gamma: gamma value
    :param reward_steps: steps of reward
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))

    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_t = net(last_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma**reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(device)

    return states_t, actions_t, ref_vals_t


def main(use_async: bool = False):

    device = torch.device(get_device())

    env = gym.make_vec("PongNoFrameskip-v4", NUM_ENVS, vectorization_mode="async" , wrappers=[wrap_dqn], render_mode="rgb_array")

    writer = SummaryWriter(comment="-pong-a2c")

    net = AtariA2C(env.single_observation_space.shape, env.single_action_space.n).to(
        device
    )

    agent = PolicyAgent(lambda x: net(x)[0], preprocessor=float32_preprocessor, device=device, apply_softmax=True)
    # Experience source that returns the first and last states only
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS
    )

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = 0

    with RewardTracker(writer, stop_reward=18) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # get new rewards from the environment
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_t, actions_t, vals_ref_t = unpack_batch(
                    batch, net, device=device, gamma=GAMMA, reward_steps=REWARD_STEPS
                )
                batch.clear()

                optimizer.zero_grad()

                logits_t, value_t = net(states_t)
                # MSE loss between the reference values and the value of the state
                # the ref values are taken from the Bellman equation
                loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)
                log_prob_t = F.log_softmax(logits_t, dim=1)
                # calculate advantage as the difference between the reference value and the value of the state
                # aka Q(s, a) - V(s) - or the rewards - the value of the state
                adv_t = vals_ref_t - value_t.detach()

                # get the log probability of the actions taken
                log_act_t = log_prob_t[range(BATCH_SIZE), actions_t]
                # multiply the log probability by the advantage
                log_prob_actions_t = adv_t * log_act_t
                # calculate the loss as the negative mean of the log probability of the actions taken
                loss_policy_t = -log_prob_actions_t.mean()

                # entropy bonus to encourage exploration
                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = ENTROPY_BETA * (prob_t * log_prob_t).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_t.backward(retain_graph=True)
                # get the grads for logging purposes
                grads = np.concatenate(
                    [
                        p.grad.data.cpu().numpy().flatten()
                        for p in net.parameters()
                        if p.grad is not None
                    ]
                )

                # apply entropy and value gradients
                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_t

                tb_tracker.track("advantage", adv_t, step_idx)
                tb_tracker.track("values", value_t, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track("loss_policy", loss_policy_t, step_idx)
                tb_tracker.track("loss_value", loss_value_t, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track(
                    "grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx
                )
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)

                best_reward_in_batch = np.mean(new_rewards)
                if best_reward_in_batch > best_reward:
                    logger.info(f"Best reward updated: {best_reward} -> {best_reward_in_batch}")
                    best_reward = best_reward_in_batch
                    ensure_directory("checkpoints")
                    torch.save(net.state_dict(), f"checkpoints/pong_a2c_{step_idx}.pt")


if __name__ == "__main__":
    fire.Fire(main)
