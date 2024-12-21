import abc
import copy
import typing as tt
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import gymnasium as gym
from dataclasses import dataclass
from collections import deque
import time

CPU_DEVICE = torch.device("cpu")
States = tt.List[np.ndarray] | np.ndarray
AgentStates = tt.List[tt.Any]
Preprocessor = tt.Callable[[States], torch.Tensor]


State = np.ndarray
Action = int


class ActionSelector(abc.ABC):
    """
    Abstract class which converts scores to the actions
    """

    @abc.abstractmethod
    def __call__(self, scores: np.ndarray) -> np.ndarray: ...


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(
        self, epsilon: float = 0.05, selector: Optional[ActionSelector] = None
    ):
        self._epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if value < 0.0 or value > 1.0:
            raise ValueError("Epsilon has to be between 0 and 1")
        self._epsilon = value

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        assert len(scores.shape) == 2
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class BaseAgent(abc.ABC):
    """
    Base Agent, sharing most of logic with concrete agent implementations.
    """

    def initial_state(self) -> tt.Optional[tt.Any]:
        """
        Should create initial empty state for the agent. It will be
        called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    @abc.abstractmethod
    def __call__(
        self, states: States, agent_states: AgentStates
    ) -> tt.Tuple[np.ndarray, AgentStates]: ...


class NNAgent(BaseAgent):
    """
    Network-based agent
    """

    def __init__(
        self,
        model: nn.Module,
        action_selector: actions.ActionSelector,
        device: torch.device,
        preprocessor: Preprocessor,
    ):
        """
        Constructor of base agent
        :param model: model to be used
        :param action_selector: action selector
        :param device: device for tensors
        :param preprocessor: states preprocessor
        """
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.preprocessor = preprocessor

    @abc.abstractmethod
    def _net_filter(
        self, net_out: tt.Any, agent_states: AgentStates
    ) -> tt.Tuple[torch.Tensor, AgentStates]:
        """
        Internal method, processing network output and states into selector's input and new states
        :param net_out: output from the network
        :param agent_states: agent states
        :return: tuple with tensor to be fed into selector and new states
        """
        ...

    @torch.no_grad()
    def __call__(
        self, states: States, agent_states: AgentStates = None
    ) -> tt.Tuple[np.ndarray, AgentStates]:
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_v = self.model(states)
        q_v, new_states = self._net_filter(q_v, agent_states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, new_states


def default_states_preprocessor(states: States) -> torch.Tensor:
    """
    Convert list of states into the form suitable for model
    :param states: list of numpy arrays with states or numpy array
    :return: torch.Tensor
    """
    if isinstance(states, list):
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.asarray([np.asarray(s) for s in states])
    else:
        np_states = states
    return torch.as_tensor(np_states)


class DQNAgent(NNAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(
        self,
        model: nn.Module,
        action_selector: ActionSelector,
        device: torch.device = CPU_DEVICE,
        preprocessor: Preprocessor = default_states_preprocessor,
    ):
        super().__init__(
            model,
            action_selector=action_selector,
            device=device,
            preprocessor=preprocessor,
        )

    # not needed in DQN - we don't process Q-values returned
    def _net_filter(
        self, net_out: tt.Any, agent_states: AgentStates
    ) -> tt.Tuple[torch.Tensor, AgentStates]:
        assert torch.is_tensor(net_out)
        return net_out, agent_states


@dataclass(frozen=True)
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    Item = tt.Tuple[Experience, ...]

    def __init__(self, env: gym.Env | tt.Collection[gym.Env], agent: BaseAgent,
                 steps_count: int = 2, steps_delta: int = 1,
                 env_seed: tt.Optional[int] = None):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param env_seed: seed to be used in Env.reset() call
        """
        assert steps_count >= 1
        if isinstance(env, (list, tuple)):
            self.pool = env
            # do the check for the multiple copies passed
            ids = set(id(e) for e in env)
            if len(ids) < len(env):
                raise ValueError("You passed single environment instance multiple times")
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.agent_states = [agent.initial_state() for _ in self.pool]
        self.env_seed = env_seed

    def __iter__(self) -> tt.Generator[Item, None, None]:
        states, histories, cur_rewards, cur_steps = [], [], [], []
        for env in self.pool:
            if self.env_seed is not None:
                obs, _ = env.reset(seed=self.env_seed)
            else:
                obs, _ = env.reset()
            states.append(obs)
            histories.append(deque(maxlen=self.steps_count))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        iter_idx = 0
        while True:
            actions, self.agent_states = self.agent(states, self.agent_states)
            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx]
                history = histories[idx]
                next_state, r, is_done, is_tr, _ = env.step(action)
                cur_rewards[idx] += r
                cur_steps[idx] += 1
                history.append(Experience(state=state, action=action, reward=r, done_trunc=is_done or is_tr))
                if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                    yield tuple(history)
                states[idx] = next_state
                if is_done or is_tr:
                    # generate tail of history
                    if 0 < len(history) < self.steps_count:
                        yield tuple(history)
                    while len(history) > 1:
                        history.popleft()
                        yield tuple(history)
                    self.total_rewards.append(cur_rewards[idx])
                    self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                    if self.env_seed is not None:
                        states[idx], _ = env.reset(seed=self.env_seed)
                    else:
                        states[idx], _ = env.reset()
                    self.agent_states[idx] = self.agent.initial_state()
                    history.clear()
            iter_idx += 1

    def pop_total_rewards(self) -> tt.List[float]:
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self) -> tt.List[tt.Tuple[float, int]]:
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


@dataclass(frozen=True)
class ExperienceFirstLast:
    state: State
    action: Action
    reward: float
    last_state: tt.Optional[State]


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env: gym.Env, agent: BaseAgent, gamma: float,
                 steps_count: int = 1, steps_delta: int = 1, env_seed: tt.Optional[int] = None):
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, env_seed=env_seed)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self) -> tt.Generator[ExperienceFirstLast, None, None]:
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done_trunc and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)

class ExperienceReplayBuffer:
    def __init__(self, experience_source: tt.Optional[ExperienceSource], buffer_size: int):
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer: tt.List[ExperienceSource.Item] = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self) -> tt.Iterator[ExperienceSource.Item]:
        return iter(self.buffer)

    def sample(self, batch_size: int) -> tt.List[ExperienceSource.Item]:
        """
        Get one random batch from experience replay
        :param batch_size: size of the batch to sample
        :return: list of experience entries
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample: ExperienceSource.Item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples: int):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

