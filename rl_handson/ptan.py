import abc
import copy
import time
import typing as tt
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine import Engine, EventEnum, Events, State
from ignite.handlers.timing import Timer
from torch import nn
from loguru import logger

CPU_DEVICE = torch.device("cpu")
States = tt.List[np.ndarray] | np.ndarray
AgentStates = tt.List[tt.Any]
Preprocessor = tt.Callable[[States], torch.Tensor]


StateType = np.ndarray
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
        action_selector: ActionSelector,
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
    state: StateType
    action: Action
    reward: float
    done_trunc: bool


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """

    Item = tt.Tuple[Experience, ...]

    def __init__(
        self,
        env: gym.Env | tt.Collection[gym.Env],
        agent: BaseAgent,
        steps_count: int = 2,
        steps_delta: int = 1,
        env_seed: tt.Optional[int] = None,
    ):
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
                raise ValueError(
                    "You passed single environment instance multiple times"
                )
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
                history.append(
                    Experience(
                        state=state,
                        action=action,
                        reward=r,
                        done_trunc=is_done or is_tr,
                    )
                )
                if (
                    len(history) == self.steps_count
                    and iter_idx % self.steps_delta == 0
                ):
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
    state: StateType
    action: Action
    reward: float
    last_state: tt.Optional[StateType]


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """

    def __init__(
        self,
        env: gym.Env | tt.Collection[gym.Env],
        agent: BaseAgent,
        gamma: float,
        steps_count: int = 1,
        steps_delta: int = 1,
        env_seed: tt.Optional[int] = None,
    ):
        super(ExperienceSourceFirstLast, self).__init__(
            env, agent, steps_count + 1, steps_delta, env_seed=env_seed
        )
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
            yield ExperienceFirstLast(
                state=exp[0].state,
                action=exp[0].action,
                reward=total_reward,
                last_state=last_state,
            )


class ExperienceReplayBuffer:
    def __init__(
        self, experience_source: tt.Optional[ExperienceSource], buffer_size: int
    ):
        self.experience_source_iter = (
            None if experience_source is None else iter(experience_source)
        )
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


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class EpsilonTracker:
    """
    Track the epsilon value for the agent,
    and update a selector accordingly
    """

    def __init__(
        self,
        selector: EpsilonGreedyActionSelector,
        epsilon_start: float,
        epsilon_final: float,
        epsilon_frames: int,
    ):
        self.selector = selector
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.epsilon_start - frame_idx / self.epsilon_frames
        self.selector.epsilon = max(self.epsilon_final, eps)


class PeriodEvents(EventEnum):
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"


class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """

    INTERVAL_TO_EVENT = {
        10: PeriodEvents.ITERS_10_COMPLETED,
        100: PeriodEvents.ITERS_100_COMPLETED,
        1000: PeriodEvents.ITERS_1000_COMPLETED,
        10000: PeriodEvents.ITERS_10000_COMPLETED,
        100000: PeriodEvents.ITERS_100000_COMPLETED,
    }

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*PeriodEvents)
        for e in PeriodEvents:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        for period, event in self.INTERVAL_TO_EVENT.items():
            if engine.state.iteration % period == 0:
                engine.fire_event(event)


class EpisodeEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EpisodeFPSHandler:
    FPS_METRIC = "fps"
    AVG_FPS_METRIC = "avg_fps"
    TIME_PASSED_METRIC = "time_passed"

    def __init__(self, fps_mul: float = 1.0, fps_smooth_alpha: float = 0.98):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()
        self._fps_smooth_alpha = fps_smooth_alpha

    def attach(self, engine: Engine, manual_step: bool = False):
        self._timer.attach(
            engine, step=None if manual_step else Events.ITERATION_COMPLETED
        )
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)
        engine.state.metrics[self.AVG_FPS_METRIC] = 0

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every time we've communicated with environment
        to get proper FPS
        :return:
        """
        self._timer.step()

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            fps = self._fps_mul / t_val
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None or avg_fps <= 0:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smooth_alpha
                avg_fps += (1 - self._fps_smooth_alpha) * fps
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts
        self._timer.reset()


class EndOfEpisodeHandler:
    def __init__(
        self,
        exp_source: ExperienceSource,
        alpha: float = 0.98,
        bound_avg_reward: Optional[float] = None,
        subsample_end_of_episode: Optional[int] = None,
    ):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param subsample_end_of_episode: if given, end of episode event will be subsampled by this amount
        """
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._best_avg_reward = None
        self._subsample_end_of_episode = subsample_end_of_episode

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics["reward"] = reward
            engine.state.metrics["steps"] = steps
            self._update_smoothed_metrics(engine, reward, steps)
            if (
                self._subsample_end_of_episode is None
                or engine.state.episode % self._subsample_end_of_episode == 0
            ):
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if (
                self._bound_avg_reward is not None
                and engine.state.metrics["avg_reward"] >= self._bound_avg_reward
            ):
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics["avg_reward"]
            elif self._best_avg_reward < engine.state.metrics["avg_reward"]:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics["avg_reward"]

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(("avg_reward", "avg_steps"), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1 - self._alpha) * val


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


def float32_preprocessor(states: States):
    np_states = np.array(states, dtype=np.float32)
    return torch.as_tensor(np_states)


class PolicyAgent(NNAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """

    def __init__(
        self,
        model: nn.Module,
        action_selector: ActionSelector = ProbabilityActionSelector(),
        device: torch.device = CPU_DEVICE,
        apply_softmax: bool = False,
        preprocessor: Preprocessor = default_states_preprocessor,
    ):
        super().__init__(
            model=model,
            action_selector=action_selector,
            device=device,
            preprocessor=preprocessor,
        )
        self.apply_softmax = apply_softmax

    def _net_filter(
        self, net_out: tt.Any, agent_states: AgentStates
    ) -> tt.Tuple[torch.Tensor, AgentStates]:
        assert torch.is_tensor(net_out)
        if self.apply_softmax:
            return F.softmax(net_out, dim=1), agent_states
        return net_out, agent_states


def vector_rewards(
    rewards: tt.Deque[np.ndarray], dones: tt.Deque[np.ndarray], gamma: float
) -> np.ndarray:
    """
    Calculate rewards from vectorized environment for given amount of steps.
    :param rewards: deque with observed rewards
    :param dones: deque with bool flags indicating end of episode
    :param gamma: gamma constant
    :return: vector with accumulated rewards
    """
    res = np.zeros(rewards[0].shape[0], dtype=np.float32)
    for r, d in zip(reversed(rewards), reversed(dones)):
        res *= gamma * (1.0 - d)
        res += r
    return res


class VectorExperienceSourceFirstLast(ExperienceSource):
    """
    ExperienceSourceFirstLast which supports VectorEnv from Gymnasium.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        agent: BaseAgent,
        gamma: float,
        steps_count: int = 1,
        env_seed: tt.Optional[int] = None,
        unnest_data: bool = True,
    ):
        """
        Construct vectorized version of ExperienceSourceFirstLast
        :param env: vectorized environment
        :param agent: agent to use
        :param gamma: gamma for reward calculation
        :param steps_count: count of steps
        :param env_seed: seed for environments reset
        :param unnest_data: should we unnest data in the iterator. If True (default)
        ExperienceFirstLast will be yielded sequentially. If False, we'll keep them in a list as we
        got them from env vector.
        """
        super().__init__(env, agent, steps_count + 1, steps_delta=1, env_seed=env_seed)
        self.env = env
        self.gamma = gamma
        self.steps = steps_count
        self.unnest_data = unnest_data
        self.agent_state = self.agent_states[0]

    def _iter_env_idx_obs_next(
        self, b_obs, b_next_obs
    ) -> tt.Generator[tt.Tuple[int, tt.Any, tt.Any], None, None]:
        """
        Iterate over individual environment observations and next observations.
        Take into account Tuple observation space (which is handled specially in Vectorized envs)
        :param b_obs: vectorized observations
        :param b_next_obs: vectorized next observations
        :yields: Tuple of index, observation and next observation
        """
        if isinstance(self.env.single_observation_space, gym.spaces.Tuple):
            obs_iter = zip(*b_obs)
            next_obs_iter = zip(*b_next_obs)
        else:
            obs_iter = b_obs
            next_obs_iter = b_next_obs
        yield from zip(range(self.env.num_envs), obs_iter, next_obs_iter)

    def __iter__(
        self,
    ) -> tt.Generator[tt.List[ExperienceFirstLast] | ExperienceFirstLast, None, None]:
        q_states = deque(maxlen=self.steps + 1)
        q_actions = deque(maxlen=self.steps + 1)
        q_rewards = deque(maxlen=self.steps + 1)
        q_dones = deque(maxlen=self.steps + 1)
        total_rewards = np.zeros(self.env.num_envs, dtype=np.float32)
        total_steps = np.zeros_like(total_rewards, dtype=np.int64)

        if self.env_seed is not None:
            obs, _ = self.env.reset(seed=self.env_seed)
        else:
            obs, _ = self.env.reset()
        env_indices = np.arange(self.env.num_envs)

        while True:
            q_states.append(obs)
            actions, self.agent_state = self.agent(obs, self.agent_state)
            q_actions.append(actions)
            next_obs, r, is_done, is_tr, _ = self.env.step(actions)
            total_rewards += r
            total_steps += 1
            done_or_tr = is_done | is_tr
            q_rewards.append(r)
            q_dones.append(done_or_tr)

            # process environments which are done at this step
            if done_or_tr.any():
                indices = env_indices[done_or_tr]
                self.total_steps.extend(total_steps[indices])
                self.total_rewards.extend(total_rewards[indices])
                total_steps[indices] = 0
                total_rewards[indices] = 0.0

            if len(q_states) == q_states.maxlen:
                # enough data for calculation
                results = []
                rewards = vector_rewards(q_rewards, q_dones, self.gamma)
                for i, e_obs, e_next_obs in self._iter_env_idx_obs_next(
                    q_states[0], next_obs
                ):
                    # if anywhere in the trajectory we have ended episode flag,
                    # the last state will be None
                    ep_ended = any(map(lambda d: d[i], q_dones))
                    last_state = e_next_obs if not ep_ended else None
                    results.append(
                        ExperienceFirstLast(
                            state=e_obs,
                            action=q_actions[0][i],
                            reward=rewards[i],
                            last_state=last_state,
                        )
                    )
                if self.unnest_data:
                    yield from results
                else:
                    yield results
            obs = next_obs


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """

    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(
            value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)
        ) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()
