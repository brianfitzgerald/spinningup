"""
Monte-Carlo Tree Search
"""

import typing as tt
import math as m
import numpy as np
import torch
import torch.nn.functional as F

from game import ConnectFour


def _encode_list_state(
    dest_np: np.ndarray,
    state_list: tt.List[tt.List[int]],
    who_move: int,
    game_rows: int,
    game_cols: int,
    obs_shape: tuple[int, int, int],
):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
    """
    assert dest_np.shape == obs_shape

    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = game_rows - rev_row_idx - 1
            if cell == who_move:
                dest_np[0, row_idx, col_idx] = 1.0
            else:
                dest_np[1, row_idx, col_idx] = 1.0


def state_lists_to_batch(
    state_lists: tt.List[tt.List[tt.List[int]]],
    who_moves_lists: tt.List[int],
    game: ConnectFour,
    device: str = "cpu",
    obs_shape: tuple[int, int, int] = (2, 6, 7),
):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
    for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
        _encode_list_state(batch[idx], state, who_move, game.rows, game.cols, obs_shape)
    return torch.tensor(batch).to(device)


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """

    def _encode_list_state(
        self, dest_np: np.ndarray, state_list: tt.List[tt.List[int]], who_move: int
    ):
        """
        In-place encodes list state into the zero numpy array
        :param dest_np: dest array, expected to be zero
        :param state_list: state of the game in the list form
        :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
        """
        assert dest_np.shape == OBS_SHAPE

        for col_idx, col in enumerate(state_list):
            for rev_row_idx, cell in enumerate(col):
                row_idx = game.GAME_ROWS - rev_row_idx - 1
                if cell == who_move:
                    dest_np[0, row_idx, col_idx] = 1.0
                else:
                    dest_np[1, row_idx, col_idx] = 1.0

    def __init__(self, game: ConnectFour, c_puct: float = 1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count: tt.Dict[int, tt.List[int]] = {}
        # total value of the state's act, state_int -> [W(s, a)]
        self.value: tt.Dict[int, tt.List[float]] = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg: tt.Dict[int, tt.List[float]] = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs: tt.Dict[int, tt.List[float]] = {}
        self.game = game

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state_int: int, player: int):
        """
        Traverse the tree until the end of game or leaf node
        :param state_int: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state_int of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * self.game.cols)
                probs = [
                    0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)
                ]
            score = [
                value + self.c_puct * prob * total_sqrt / (1 + count)
                for value, prob, count in zip(values_avg, probs, counts)
            ]
            invalid_actions = set(range(self.game.cols)) - set(
                self.game.possible_moves(cur_state)
            )
            for invalid in invalid_actions:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            actions.append(action)
            cur_state, won = self.game.move(cur_state, action, cur_player)
            if won:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0
            cur_player = 1 - cur_player
            # check for the draw
            moves_count = len(self.game.possible_moves(cur_state))
            if value is None and moves_count == 0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def search_batch(self, count, batch_size, state_int, player, net, device="cpu"):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net, device)

    def search_minibatch(self, count, state_int, player, net, device="cpu"):
        """
        Perform several MCTS searches.
        """
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(
                state_int, player
            )
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_lists = self.game.decode_binary(leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states, actions))

        # do expansion of nodes
        if expand_queue:
            batch_v = state_lists_to_batch(
                expand_states, expand_players, self.game, device
            )
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in zip(
                expand_queue, values, probs
            ):
                self.visit_count[leaf_state] = [0] * self.game.cols
                self.value[leaf_state] = [0.0] * self.game.cols
                self.value_avg[leaf_state] = [0.0] * self.game.cols
                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] = (
                    self.value[state_int][action] / self.visit_count[state_int][action]
                )
                cur_value = -cur_value

    def get_policy_value(self, state_int, tau: float, n_cols: int):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :param tau: temperature parameter
        :param n_cols: number of columns in the game
        :return: (probs, values)
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * n_cols
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values
