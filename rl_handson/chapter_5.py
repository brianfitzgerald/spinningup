"""
Tabular learning and Q-learning
Represent the environment as a Markov chain with state transitions
Create a table of Q-values for each state-action pair

Bellman equation: the optimal Q-value for a state-action pair is the sum of the immediate reward and the discounted future reward
Sum this for all possible actions and take the maximum

Q-function: the total reward we can get from execution action a in state s, defined as V(s)
Q(s, a) = r + gamma * max(Q(s', a'))
Calculate q-values for all possible actions and take the maximum
This can be seen as a recursive function, and can be solved via dynamic programming

One issue is loops - to solve, use a value iteration algorithm
Numerically calculate the value of states and actions for N steps

For continuous problems, like CartPole, can bin the state space into discrete states
Or use a function approximator, like a neural network, to estimate the Q-values

Also need to estimate the probability of a transition, as well as the reward for each transition

"""

import fire
import gymnasium
from tensorboardX import SummaryWriter
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from collections import defaultdict, Counter


class Agent:
    def __init__(self, env: Env):
        self.env = env
        # environment state
        self.state = self.env.reset()
        # table of reward estimates
        self.rewards = defaultdict(float)
        # table of state transition counts
        self.transits = defaultdict(Counter)
        # table of value estimates
        self.values = defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def play_episode(self):
        total_reward = 0.0
        state, _ = self.env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


def main():

    env = gymnasium.make(ENV_NAME)
    env = RecordVideo(env, video_folder=f"videos/chapter_5")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()


fire.Fire(main)
