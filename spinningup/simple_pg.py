import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box
import fire
from typing import List
from gymnasium.wrappers import RecordVideo, TimeLimit


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def reward_to_go(rewards):
    """
    Discount rewards to only consider the rewards that come after the current time step.
    """
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(
    env_name="LunarLander-v2",
    hidden_sizes: List[int] = [32, 64, 32],
    lr: float = 1e-2,
    epochs: int = 250,
    batch_size: int = 5000,
    render: bool = False,
    rtg: bool = True,
):
    # make environment, check spaces, get obs / act dims
    render_mode = "human" if render else "rgb_array"
    base_env = gymnasium.make(env_name, render_mode=render_mode)
    base_env = TimeLimit(base_env, max_episode_steps=200)
    record_env = RecordVideo(base_env, f"videos/{env_name}")
    assert isinstance(
        base_env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        base_env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = base_env.observation_space.shape[0]
    n_acts = base_env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch(env: gymnasium.Env):
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                if rtg:
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, _ = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(base_env)
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )
        if i % 50 == 0:
            train_one_epoch(record_env)

    base_env.close()


if __name__ == "__main__":
    fire.Fire(train)
