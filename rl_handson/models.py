from typing import Tuple
import torch
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from lib_textworld import unpack_batch
from ptan import (
    AgentStates,
    BaseAgent,
    States,
    float32_preprocessor,
)
import numpy as np


class DQNConvNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int], n_actions: int):
        super(DQNConvNet, self).__init__()

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


class DQNLinear(nn.Module):
    def __init__(self, obs_size: int, cmd_size: int, hid_size: int = 256):
        super(DQNLinear, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + cmd_size, hid_size), nn.ReLU(), nn.Linear(hid_size, 1)
        )

    def forward(self, obs, cmd):
        x = torch.cat((obs, cmd), dim=1)
        return self.net(x)

    @torch.no_grad()
    def q_values(self, obs_t: torch.Tensor, commands_t: torch.Tensor):
        """
        Calculate q-values for observation and tensor of commands
        :param obs_t: preprocessed observation, need to be of [1, obs_size] shape
        :param commands_t: commands to be evaluated, shape is [N, cmd_size]
        :return: list of q-values for commands
        """
        result = []
        for cmd_t in commands_t:
            qval = self(obs_t, cmd_t.unsqueeze(0))[0].cpu().item()
            result.append(qval)
        return result

    @torch.no_grad()
    def q_values_cmd(self, obs_t: torch.Tensor, commands_t: torch.Tensor):
        x = torch.cat(torch.broadcast_tensors(obs_t.unsqueeze(0), commands_t), dim=1)
        q_vals = self.net(x)
        return q_vals.cpu().numpy()[:, 0].tolist()


class SimpleLinear(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(SimpleLinear, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AtariA2C(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super(AtariA2C, self).__init__()

        # Convolutional layers, returns feature maps of size 64
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        logger.info(f"Feature size: {size}")
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )
        # Value network
        self.value = nn.Sequential(nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x: torch.ByteTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xx = x / 255
        conv_out = self.conv(xx)
        return self.policy(conv_out), self.value(conv_out)


def calc_loss_dqn(
    batch, preprocessor, tgt_preprocessor, net, tgt_net, gamma, device="cpu"
):
    states, taken_commands, rewards, next_best_qs = unpack_batch(
        batch, tgt_preprocessor, tgt_net, device
    )

    obs_t = preprocessor.encode_observations(states).to(device)
    cmds_t = preprocessor.encode_commands(taken_commands).to(device)
    q_values_t = net(obs_t, cmds_t)
    tgt_q_t = torch.tensor(rewards) + gamma * torch.tensor(next_best_qs)
    tgt_q_t = tgt_q_t.to(device)
    return F.mse_loss(q_values_t.squeeze(-1), tgt_q_t)


class ModelSACTwinQ(nn.Module):
    """
    Twin Q-network for SAC
    Give 2 different Q estimations, and take the lower of the two
    """

    def __init__(self, obs_size: int, act_size: int, hid_size: int):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)


class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    OU generates noise that is correlated with the previous noise,
    adding a form of momentum to the exploration process, which can aid in training stability.
    Also helpful for environments that have inertia, such as physical control
    """

    def __init__(
        self,
        net: nn.Module,
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
        # Speed of mean reversion
        self.ou_teta = ou_teta
        # Noise volatility
        self.ou_sigma = ou_sigma
        # Noise scale
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
