import math as m
import typing as tt
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from game import BaseGame

# reuse function as net input matches
from mcts import state_lists_to_batch

Action = int


@dataclass
class EpisodeStep:
    state: int
    player_idx: int
    action: int
    reward: int


@dataclass
class MuZeroParams:
    # derived from game params
    actions_count: int
    max_moves: int
    device: str

    dirichlet_alpha: float = 0.3
    discount: float = 1.0
    # how long hidden states dynamics are unrolled
    unroll_steps: int = 5

    # UCB formula
    pb_c_base: int = 19652
    pb_c_init: float = 1.25


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = -1.0
        self.minimum = 1.0

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        return (value - self.minimum) / (self.maximum - self.minimum)


class MCTSNode:
    def __init__(self, prior: float, first_plays: bool):
        self.first_plays: bool = first_plays
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: tt.Dict[Action, MCTSNode] = {}
        # node is not expanded, so has no hidden state
        # This is an encoded hidden state, encoded by the representation model
        self.h = None
        # predicted reward
        self.r = 0.0

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return 0 if not self.visit_count else self.value_sum / self.visit_count

    def select_child(
        self, params: MuZeroParams, min_max: MinMaxStats
    ) -> tt.Tuple[Action, "MCTSNode"]:
        max_ucb, best_action, best_node = None, None, None
        for action, node in self.children.items():
            ucb = ucb_value(params, self, node, min_max)
            if max_ucb is None or max_ucb < ucb:
                max_ucb = ucb
                best_action = action
                best_node = node
        assert best_action is not None
        assert best_node is not None
        return best_action, best_node

    def get_act_probs(self, t: float = 1) -> tt.List[float]:
        child_visits = sum(map(lambda n: n.visit_count, self.children.values()))
        p = np.array(
            [
                (child.visit_count / child_visits) ** (1 / t)
                for _, child in sorted(self.children.items())
            ]
        )
        p /= sum(p)
        return list(p)

    def select_action(self, t: float, params: MuZeroParams) -> Action:
        """
        Select action from visit counts using softmax with temperature.
        :param t: temperature to be used (from 0.00001 to inf)
        :return: sampled action
        """
        act_vals = list(sorted(self.children.keys()))

        if not act_vals:
            res = np.random.choice(params.actions_count)
        elif t < 0.0001:
            res, _ = max(self.children.items(), key=lambda p: p[1].visit_count)
        else:
            p = self.get_act_probs(t)
            res = int(np.random.choice(act_vals, p=p))
        return res


def ucb_value(
    params: MuZeroParams, parent: MCTSNode, child: MCTSNode, min_max: MinMaxStats
) -> float:
    pb_c = (
        m.log((parent.visit_count + params.pb_c_base + 1) / params.pb_c_base)
        + params.pb_c_init
    )
    pb_c *= m.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = 0.0
    if child.visit_count > 0:
        value_score = min_max.normalize(child.value + child.r)
    return prior_score + value_score


class Episode:
    def __init__(self):
        self.steps: tt.List[EpisodeStep] = []
        self.action_probs: tt.List[tt.List[float]] = []
        self.root_values: tt.List[float] = []

    def __len__(self):
        return len(self.steps)

    def add_step(self, step: EpisodeStep, node: MCTSNode):
        self.steps.append(step)
        self.action_probs.append(node.get_act_probs())
        self.root_values.append(node.value)


class ReprModel(nn.Module):
    """
    Representation model, maps observations into the hidden state
    """

    def __init__(
        self, input_shape: Tuple[int, ...], num_filters: int, hidden_size: int
    ):
        super(ReprModel, self).__init__()
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
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_filters, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        body_shape = (num_filters,) + input_shape[1:]
        size = self.conv_out(torch.zeros(1, *body_shape)).size()[-1]
        self.out = nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )

    def forward(self, x):
        v = self.conv_in(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        c_out = self.conv_out(v)
        out = self.out(c_out)
        return out


class PredModel(nn.Module):
    """
    Prediction model, maps hidden state into policy and value
    """

    def __init__(self, actions: int, hidden_size: int):
        super(PredModel, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns policy and value from the hidden state
        """
        return self.policy(x), self.value(x).squeeze(1)


class DynamicsModel(nn.Module):
    """
    Dynamics model, maps hidden state and action into
    reward and new hidden state.
    """

    def __init__(self, actions: int, hidden_size: int):
        super(DynamicsModel, self).__init__()
        self.reward = nn.Sequential(
            nn.Linear(hidden_size + actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.hidden = nn.Sequential(
            nn.Linear(hidden_size + actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )

    def forward(
        self, h: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of dynamics model
        :param h: batch of hidden states
        :param a: batch of one-hot actions
        :return: predicted rewards and new hidden states
        """
        x = torch.hstack((h, a))
        return self.reward(x).squeeze(1), self.hidden(x)


class MuZeroModels:
    def __init__(self, input_shape: Tuple[int, ...], actions: int, hidden_size: int):
        self.repr = ReprModel(input_shape, 64, hidden_size)
        self.pred = PredModel(actions, hidden_size)
        self.dynamics = DynamicsModel(actions, hidden_size)

    def to(self, dev: torch.device):
        self.repr.to(dev)
        self.pred.to(dev)
        self.dynamics.to(dev)

    def sync(self, src: "MuZeroModels"):
        self.repr.load_state_dict(src.repr.state_dict())
        self.pred.load_state_dict(src.pred.state_dict())
        self.dynamics.load_state_dict(src.dynamics.state_dict())

    def get_state_dict(self) -> Dict[str, dict]:
        return {
            "repr": self.repr.state_dict(),
            "pred": self.pred.state_dict(),
            "dynamics": self.dynamics.state_dict(),
        }

    def set_state_dict(self, d: dict):
        self.repr.load_state_dict(d["repr"])
        self.pred.load_state_dict(d["pred"])
        self.dynamics.load_state_dict(d["dynamics"])


def backpropagate(
    search_path: tt.List[MCTSNode],
    value: float,
    first_plays: bool,
    params: MuZeroParams,
    min_max: MinMaxStats,
):
    for node in reversed(search_path):
        node.value_sum += value if node.first_plays == first_plays else -value
        node.visit_count += 1
        value = node.r + params.discount * value
        min_max.update(value)


def make_expanded_root(
    player_idx: int,
    game_state_int: int,
    params: MuZeroParams,
    models: MuZeroModels,
    min_max: MinMaxStats,
    game: BaseGame,
) -> MCTSNode:
    root = MCTSNode(1.0, player_idx == 0)
    state_list = game.decode_binary(game_state_int)
    state_t = state_lists_to_batch(
        [state_list], [player_idx], game, device=params.device
    )
    h_t = models.repr(state_t)
    root.h = h_t[0].cpu().numpy()

    p_t, v_t = models.pred(h_t)
    # logits to probs
    p_t.exp_()
    probs_t = p_t.squeeze(0) / p_t.sum()
    probs = probs_t.cpu().numpy()
    # add dirichlet noise
    noises = np.random.dirichlet([params.dirichlet_alpha] * params.actions_count)
    probs = probs * 0.75 + noises * 0.25
    for a, prob in enumerate(probs):
        root.children[a] = MCTSNode(prob, not root.first_plays)
    # backpropagate value
    v = v_t.cpu().item()
    backpropagate([root], v, root.first_plays, params, min_max)
    return root


def expand_node(
    parent: MCTSNode,
    node: MCTSNode,
    last_action: Action,
    params: MuZeroParams,
    models: MuZeroModels,
) -> float:
    """
    Performs node expansion using models.
    Return predicted value for the node's state.
    :param parent: parent's node
    :param node: node to be expanded
    :param action: action from the parent's node
    :param params: hyperparams
    :param models: models
    :return: predicted value to be backpropagated
    """
    h_t = torch.as_tensor(parent.h, dtype=torch.float32, device=params.device)
    h_t.unsqueeze_(0)
    p_t, v_t = models.pred(h_t)
    # one-hot of actions
    a_t = torch.zeros(params.actions_count, dtype=torch.float32, device=params.device)
    a_t[last_action] = 1.0
    a_t.unsqueeze_(0)
    # predict the reward and the next hidden state
    r_t, h_next_t = models.dynamics(h_t, a_t)
    node.h = h_next_t[0].cpu().numpy()
    node.r = float(r_t[0].cpu().item())

    # convert logits to probs
    p_t.squeeze_(0)
    p_t.exp_()
    probs_t = p_t / p_t.sum()
    probs = probs_t.cpu().numpy()
    for a, prob in enumerate(probs):
        node.children[a] = MCTSNode(prob, not node.first_plays)
    return float(v_t.cpu().item())


@torch.no_grad()
def run_mcts(
    player_idx: int,
    root_state_int: int,
    params: MuZeroParams,
    models: MuZeroModels,
    min_max: MinMaxStats,
    game: BaseGame,
    search_rounds: int = 800,
) -> MCTSNode:
    # prepare root node
    root = make_expanded_root(player_idx, root_state_int, params, models, min_max, game)
    for _ in range(search_rounds):
        search_path = [root]
        parent_node = None
        last_action = 0  # to make type checker happy
        node = root
        while node.is_expanded:
            action, new_node = node.select_child(params, min_max)
            last_action = action
            parent_node = node
            node = new_node
            search_path.append(new_node)
        assert parent_node is not None
        value = expand_node(parent_node, node, last_action, params, models)
        backpropagate(search_path, value, node.first_plays, params, min_max)
    return root


@torch.no_grad()
def play_game(
    player1: MuZeroModels,
    player2: MuZeroModels,
    params: MuZeroParams,
    temperature: float,
    game: BaseGame,
    init_state: tt.Optional[int] = None,
) -> tt.Tuple[int, Episode]:
    episode = Episode()
    state = game.initial_state if init_state is None else init_state
    players = [player1, player2]
    player_idx = 0
    reward = 0
    min_max = MinMaxStats()

    while True:
        possible_actions = game.possible_moves(state)
        # we have a draw situation
        if not possible_actions:
            break

        root_node = run_mcts(
            player_idx, state, params, players[player_idx], min_max, game
        )
        # run_mcts(node, action, params, players[player_idx])
        action = root_node.select_action(temperature, params)

        # act randomly on wrong move
        if action not in possible_actions:
            action = int(np.random.choice(possible_actions))

        new_state, won = game.move(state, action, player_idx)
        if won:
            if player_idx == 0:
                reward = 1
            else:
                reward = -1
        step = EpisodeStep(state, player_idx, action, reward)
        episode.add_step(step, root_node)
        if won:
            break
        player_idx = (player_idx + 1) % 2
        state = new_state
    return reward, episode


def sample_batch(
    episode_buffer: tt.Deque[Episode],
    batch_size: int,
    params: MuZeroParams,
    game: BaseGame,
) -> tt.Tuple[
    torch.Tensor,
    tt.Tuple[torch.Tensor, ...],
    tt.Tuple[torch.Tensor, ...],
    tt.Tuple[torch.Tensor, ...],
    tt.Tuple[torch.Tensor, ...],
]:
    """
    Sample training batch from episode buffer
    :param episode_buffer: buffer with episodes
    :param batch_size: size of the batch
    :param params: hyperparameters
    :return: tensor with encoded states,
        tuple with one-hot actions,
        tuple with unrolled policy targets,
        tuple with unrolled rewards,
        tuple with unrolled values
    """
    states = []
    player_indices = []
    actions = [[] for _ in range(params.unroll_steps)]
    policy_targets = [[] for _ in range(params.unroll_steps)]
    rewards = [[] for _ in range(params.unroll_steps)]
    values = [[] for _ in range(params.unroll_steps)]

    for episode in np.random.choice(episode_buffer, batch_size):  # type: ignore
        assert isinstance(episode, Episode)
        ofs = np.random.choice(len(episode) - params.unroll_steps)
        state = game.decode_binary(episode.steps[ofs].state)
        states.append(state)
        player_indices.append(episode.steps[ofs].player_idx)

        for s in range(params.unroll_steps):
            full_ofs = ofs + s
            actions[s].append(episode.steps[full_ofs].action)
            rewards[s].append(episode.steps[full_ofs].reward)
            policy_targets[s].append(episode.action_probs[full_ofs])

            # compute discounted value target till the end of episode
            value = 0.0
            for step in reversed(episode.steps[full_ofs:]):
                value *= params.discount
                value += step.reward
            values[s].append(value)
    states_t = state_lists_to_batch(states, player_indices, game, device=params.device)
    res_actions = tuple(
        torch.as_tensor(
            np.eye(params.actions_count)[a], dtype=torch.float32, device=params.device
        )
        for a in actions
    )
    res_policies = tuple(
        torch.as_tensor(p, dtype=torch.float32, device=params.device)
        for p in policy_targets
    )
    res_rewards = tuple(
        torch.as_tensor(r, dtype=torch.float32, device=params.device) for r in rewards
    )
    res_values = tuple(
        torch.as_tensor(v, dtype=torch.float32, device=params.device) for v in values
    )
    return states_t, res_actions, res_policies, res_rewards, res_values
