"""
In most real world use cases, the size of the state transition table would be too large to store in memory.
Consider all possible statest of a display, like an Atari game
Also limited to discrete action spaces, instead of continuous action spaces

So the solution is to use a function approximator, like a neural network, to estimate the Q-values
Update the Q-values using the Bellman equation, with SGD
Could use the Q function approximation as a source of behavior
Epsilon-greedy method - with probability epsilon, select a random action, otherwise select the action with the highest Q-value

Since the samples aren't iid, we can't use the standard SGD
so instead use experience replay
need to be iid so that we don't bias against the most recent samples
use a replay buffer - a fixed-size buffer that stores the most recent transitions, which we can sample from

Target network trick - keep a copy of our network and use it for the Q value approximation
This network is updated less frequently than the main network, i.e. 1k or 10k steps

Partially observable MDPs - MDP without the Markov property
For games with hidden information, other players' hands need to be approximated, etc.

To solve, use a sliding window of previous states as the observation
This lets the agent deduce dynamics, of the current state

Epsilon-greedy, replay buffer, and target network are all implemented in the DQN algorithm published by DeepMind in 2015
Paper is called "Human-level control through deep reinforcement learning"

Algorithm is:
- Initialize Q(s, a) with random weights, and empty the replay buffer
- select a random action with probability epsilon, otherwise select the action with the highest Q-value
- Ecxecute the action and store the transition in the replay buffer
- Sample a random minibatch of transitions from the replay buffer
- Calculate the target Q-value using the Bellman equation
- Calculate the loss - MSE betwewen the predicted Q-value and the target Q-value
- Update Q(s, a) using SGD
- Every C steps, copy the weights of the main network to the target network
- Repeat until convergence

"""

import fire
import gymnasium
from tensorboardX import SummaryWriter
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from collections import defaultdict, Counter
import torch.nn as nn
import torch

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class DQN(nn.Module):

    def __init__(self, input_shape: tuple[int, int], n_actions: int):
        super(DQN, self).__init__()

        # convolutional layers, from image input shape to feature maps
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # get the size of the output of the conv layer
        size = self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)
        # output q values for each action
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

def main(q_iteration=True):

    env = gymnasium.make(ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=f"videos/chapter_5")
    writer = SummaryWriter(comment="-v-iteration")

    writer.close()


fire.Fire(main)
