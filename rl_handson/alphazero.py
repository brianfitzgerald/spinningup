from collections import deque
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from game import ConnectFour
from mcts import MCTS


class Net(nn.Module):
    def __init__(self, input_shape, actions_n, num_filters):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )

        # layers with residual
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
        )

        body_shape = (num_filters,) + input_shape[1:]

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        size = self.conv_val(torch.zeros(1, *body_shape)).size()[-1]
        self.value = nn.Sequential(
            nn.Linear(size, 20), nn.LeakyReLU(), nn.Linear(20, 1), nn.Tanh()
        )

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        size = self.conv_policy(torch.zeros(1, *body_shape)).size()[-1]
        self.policy = nn.Sequential(nn.Linear(size, actions_n))

    def forward(self, x):
        v = self.conv_in(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        # value output
        val = self.conv_val(v)
        val = self.value(val)
        # policy output
        pol = self.conv_policy(v)
        pol = self.policy(pol)
        return pol, val


def play_game(
    game: ConnectFour,
    mcts_stores: Optional[Union[MCTS, List[MCTS]]],
    replay_buffer: Optional[deque],
    net1: nn.Module,
    net2: nn.Module,
    steps_before_tau_0: int,
    mcts_searches: int,
    mcts_batch_size: int,
    net1_plays_first: Optional[bool] = None,
    device: str = "cpu",
):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
    if mcts_stores is None:
        mcts_stores = [MCTS(game), MCTS(game)]
    elif isinstance(mcts_stores, MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game.initial_state
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        # perform MCTS search for the current player
        mcts_stores[cur_player].search_batch(
            mcts_searches,
            mcts_batch_size,
            state,
            cur_player,
            nets[cur_player],
            device=device,
        )
        # get the probabilities from the MCTS
        probs, _ = mcts_stores[cur_player].get_policy_value(
            state, tau=tau, n_cols=game.cols
        )
        game_history.append((state, cur_player, probs))
        action = np.random.choice(game.cols, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state, won = game.move(state, action, cur_player)
        if won:
            result = 1
            net1_result = 1 if cur_player == 0 else -1
            break
        cur_player = 1 - cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            result = 0
            net1_result = 0
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    # Add game transitions into the replay buffer
    # alternate the players to be the player1 in the transitions
    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append((state, cur_player, probs, result))
            result = -result

    return net1_result, step
