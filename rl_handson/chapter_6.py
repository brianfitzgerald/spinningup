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

import time
from collections import deque
from dataclasses import dataclass, astuple
from typing import Any, List, Optional, Tuple

import ale_py
import fire
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from lib import ensure_directory, wrap_dqn
from tensorboardX import SummaryWriter
from torch.optim import Adam

gymnasium.register_envs(ale_py)

BatchTensors = Tuple[
    torch.ByteTensor,  # current state
    torch.LongTensor,  # actions
    torch.Tensor,  # rewards
    torch.BoolTensor,  # done || trunc
    torch.ByteTensor,  # next state
]


class ImageToPyTorch(ObservationWrapper):
    """
    Wrapper that converts the input image to PyTorch format (C, H, W)
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = Box(
            low=obs.low.min(), high=obs.high.max(), shape=new_shape, dtype=obs.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


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
            nn.ReLU(),
            nn.Flatten(),
        )
        # get the size of the output of the conv layer
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        # output q values for each action
        self.fc = nn.Sequential(
            nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor):
        # scale on GPU
        xx = x / 255.0
        return self.fc(self.conv(xx))


State = np.ndarray
Action = int


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env: Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
        self, net: DQN, device: torch.device, epsilon: float = 0.0
    ) -> Optional[float]:
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # Get the max Q-value and perform that action
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done_trunc=is_done or is_tr,
            new_state=new_state,
        )
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def batch_to_tensors(batch: List[Experience], device: torch.device) -> BatchTensors:
    """
    Convert a batch of experiences to tensors
    """
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return (
        states_t.to(device),
        actions_t.to(device),
        rewards_t.to(device),
        dones_t.to(device),
        new_states_t.to(device),
    )


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

# Decay epsilon from start to final in the first 10^5 frames
# This means we start with a high exploration rate, and then decrease it
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


def calc_loss(
    batch: List[Experience], net: DQN, tgt_net: DQN, device: torch.device
) -> torch.Tensor:

    # Get experiences as batch of tensors
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(
        batch, device
    )

    # get the action value pairs for the current state
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        # set the next state values to 0 if the episode is done
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    # scale the expected rewards by gamma and add to the next state values
    expected_state_action_values = next_state_values * GAMMA + rewards_t
    # calculate the loss between the predicted and expected state action values
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def get_device():
    return "cuda" if torch.cuda.is_available() else "mps"

def main(env_name: str = DEFAULT_ENV_NAME):

    device = torch.device(get_device())

    env = gymnasium.make(env_name, render_mode="rgb_array")
    env: Env = RecordVideo(env, video_folder=f"videos/chapter_5")
    # scale frame size, clip rewards, and convert to grayscale
    env = wrap_dqn(env, clip_reward=False, noop_max=0)

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    writer = SummaryWriter(comment="-v-iteration")

    while True:
        frame_idx += 1
        epsilon = max(
            EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME
        )

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(
                f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                f"eps {epsilon:.2f}, speed {speed:.2f} f/s"
            )
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                save_location = f"checkpoints/{env_name}-best_{m_reward:.0f}.dat"
                ensure_directory("checkpoints", False)
                torch.save(net.state_dict(), save_location)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()

    writer.close()


fire.Fire(main)
