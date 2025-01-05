"""
Value iteration - find the value of a state-action pair, or just the value of a state
Q learning finds the values of all state action pairs, which has downsides; it requires storing and evaluating
the value of every state-action pair, which is not feasible for large state spaces, or continuous state spaces
We do not care about the value of every action; we just want to know the best action to take in a given state

Instead, can use a policy gradient method to learn a policy directly, without estimating the value of every state-action pair
In this case, we sample from a probability distribution over actions, and update the policy to increase the probability of good actions

REINFORCE is a vanilla policy gradient method that uses the policy gradient theorem to update the policy
The theorem = gradient of the expected return with respect to the policy parameters = expected return * gradient of the log probability of the action
L = -E[log(pi(a|s)) * G], where L is the loss, pi is the policy, a is the action, s is the state, and G is the return
Negative sign because we want to maximize the return, but we minimize the loss

In cross entropy method, train on better than average episodes with a CE loss on transitions from these episodes
In REINFORCE, train on Q-estimates of actions taken in the episode
Full process:

- Initialize the policy
- Play N episodes using the current policy, saving their transitions as (s, a, r, s') tuples
- Calculate the discounted total reward for subsequent steps
- Calculate the loss as the sum of the log probabilities of the actions taken in the episode, multiplied by the Q-estimates of these actions
- Update the policy using the loss, repeat until converged

Benefit is that since we are sampling probabilities, no epsiolon-greedy exploration is needed
Nor a replay buffer since it's on-policy
No need for a target network, either

Issues with REINFORCE:
- requires full epsiodes to calculate the loss, which is inefficient
- high variance in the gradient estimates, which can lead to slow convergence, since the gradient is the sum of the rewards, which is environment depenednt
- can be unstable, since the policy can change drastically from one episode to the next
- possible to get stuck in local optima; to solve, subtract entropy from the loss, which encourages exploration
- Heavily correlated samples, which can lead to slow convergence; to solve, use a baseline, which is the average return of the episode, and subtract it from the return

"""

import gymnasium as gym
from ptan import ExperienceSourceFirstLast, PolicyAgent, float32_preprocessor
import numpy as np
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter
from models import SimpleLinear
import fire
from loguru import logger
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    """
    Calculate the rewards for each step in the episode
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    writer = SummaryWriter(comment="-cartpole-reinforce")
    env = RecordVideo(env, video_folder=f"videos/chapter_11/cartpole")

    net = SimpleLinear(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)
    # Experience source that returns the first and last states only
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        # if the episode is finished, calculate the q-values for the episode
        # which in this case are the discounted rewards for each state
        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards
        # this returns the total rewards for the episode
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            logger.info(
                f"{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, "
                f"episodes: {done_episodes}"
            )
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 450:
                print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_t = torch.as_tensor(np.asarray(batch_states))
        batch_actions_t = torch.as_tensor(np.asarray(batch_actions))
        batch_qvals_t = torch.as_tensor(np.asarray(batch_qvals))

        # get the logits for the states in the current batch
        logits_t = net(states_t)
        # softmax the logits
        log_prob_t = F.log_softmax(logits_t, dim=1)

        batch_idx = range(len(batch_states))
        # get the log probabilities of the actions taken
        act_probs_t = log_prob_t[batch_idx, batch_actions_t]
        log_prob_actions_v = batch_qvals_t * act_probs_t
        loss_t = -log_prob_actions_v.mean()

        loss_t.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()


if __name__ == "__main__":
    fire.Fire(main)
