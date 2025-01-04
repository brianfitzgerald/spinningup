"""
AlphaGo is a model-based method, in contrast with AlphaZero, which is model-free.
PPO and DQN are also model-free methods, since they learn the policy or the value function directly from the environment,
as opposed to learning a model of the environment and then using it to plan.

Model based methods are more sample efficient, and can be more data efficient, but they are also more complex and harder to train.
Model refers to a model of the environment
Once a model is learned, it can be used to plan, which can be more efficient than learning the policy or value function directly.
Then, we have less of a dependency on the real environment, and can use the model to generate more data.
Models are also transferable - we can learn a model in one environment and use it in another, or for different tasks.

AlphaGo overview:
- MCTS algorithm for traversing game states
    - core idea is to randomly walk down the game states, expand them and gather statistics about the states
    - then, use these statistics to guide the search towards the most promising states
- Use a neural network to evaluate game states during the MCTS search
- then, use self play to train the neural network, by playing games against itself

MCTS:
- branching factor is the number of possible actions at any state in the game
- Total number of possible game states is the branching factor raised to the power of the depth of the game tree
- To deal with the huge number of possible states, MCTS uses a tree search algorithm
- Depth-first search - starting at the current game state, select the most promising action, or a random action, then repeat the process from the new state
- Process is similar to value iteration, where episodes are playedr and the final step of the episode is used to update the value function

In AlphaGo:
- For every edge, store:
    - Prior probability of the edge
    - Visit count
    - Total action value
- Select an action following Q(s,a) + U(s,a) formula, where U(s,a) is the prior probability of the edge
- This is calculated as P(s, a) / 1 + N(s, a), where P(s, a) is the prior probability and N(s, a) is the visit count
- Randomness is added to the selection process by adding noise to the prior probabilities
- A neural network is used to obtain the prior probabilities, and the value of the state estimation, Q(s)
- Once the value is obtained by ending the game, or continuing the search, the backup operation is performed - the value is propagated back up the tree, though each visited intermediate node
    - We update the visit count, N(s, a), and the total action value, Q(s, a)
- This search process is performed 1-2k times, each move
- Self-play is used to train the neural network
    - Start with completely random moves, and transition to deterministic moves as the network is trained.
    - Select the action with the largest visit count
- Once self-play is finished, each step of the game is added to the training dataset
- Use minibatches from the replay buffer, filled from the self-play games, to train the neural network
- Minimize the MSE between the value head position and the actual position value, as well as the CE loss between the policy head and the actual policy

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


def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)