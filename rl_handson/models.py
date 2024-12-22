from typing import Tuple
import torch
import torch.nn as nn
from loguru import logger


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



class SimpleLinear(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(SimpleLinear, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
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
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # Value network
        self.value = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.ByteTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xx = x / 255
        conv_out = self.conv(xx)
        return self.policy(conv_out), self.value(conv_out)
