import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor as T
from gymnasium import Env
import time
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam

import gymnasium

# https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs: T):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)
        return output


class PPO:
    def __init__(self, env: Env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]  # type: ignore
        self.act_dim = env.action_space.shape[0]  # type: ignore

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.gamma = 0.95
        self.clip = 0.2        


        self.render = False
        self.render_every_i = 10
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

        self.lr = 0.005 # Learning rate of actor optimizer

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # covariance for the action matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, obs):

        # actor is the actor network
        # get the actor's mean prediction
        mean = self.actor(obs)
        # create a multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample an action from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def rollout(self):
        # Batch data
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if (
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                t += 1

                # Track observations for batch
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                # TODO args might be wrong for this
                obs, rew, terminated, truncated, _ = self.env.step(action)

                # Don't really care about the difference between terminated or truncated in this case, so just combine them
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtgs) per episode to return.
        # The shape will be (traj_len, 1), where traj_len is the length of the trajectory
        batch_rtgs = []

        # Iterate through each episode backwards to maintain correct order of rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            # get the discount reward per step
            # this is calculated by the sum of the rewards at each step, starting from the last step
            # and multiplying by the discount factor backwards
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def learn(self, total_timesteps):
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            V, _ = self.evaluate(batch_obs, batch_acts)
            # Calculate advantage for the kth iteration of the algorithm
            # detach since V should not have a gradient
            A_k = batch_rtgs - V.detach()

            # Normalize the advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            for _ in range(self.n_updates_per_iteration):
                
                # calculate V_phi and pi_theta
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # compute surrogate loss for PPO
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                # Compute loss. Take the negative min since we're maximizing the objective, but 
                # Adam minimizes the loss; so minimizing the negative objective maximizes the objective
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(V, batch_rtgs)


                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.item())

            print(self.logger)

    def evaluate(self, batch_obs, batch_acts):
        # Get the value of a specific state
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of the batch actions for the current state
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

if __name__ == "__main__":
    env = gymnasium.make("Pendulum-v1")
    ppo = PPO(FeedForwardNN, env)
    ppo.learn(100000)
    env.close()