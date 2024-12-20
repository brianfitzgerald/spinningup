from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from ptan.common.wrappers import ImageToPyTorch
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     NoopResetEnv,
                                                     StickyActionEnv,
                                                     WarpFrame)
from stable_baselines3.common.type_aliases import GymStepReturn


class JustSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        info = {}
        obs = None
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = JustSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)


class BufferWrapper(ObservationWrapper):
    """
    Wrapper that stores the last n observations in the state
    """

    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, Box)
        new_obs = Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype,
        )
        self.observation_space = new_obs
        self.buffer = deque(maxlen=n_steps)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        for _ in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def wrap_dqn(
    env: gym.Env,
    stack_frames: int = 4,
    episodic_life: bool = True,
    clip_reward: bool = True,
    noop_max: int = 0,
) -> gym.Env:
    """
    Apply a common set of wrappers for Atari games.
    :param env: Environment to wrap
    :param stack_frames: count of frames to stack, default=4
    :param episodic_life: convert life to end of episode
    :param clip_reward: reward clipping
    :param noop_max: how many NOOP actions to execute
    :return: wrapped environment
    """
    assert "NoFrameskip" in env.spec.id
    env = AtariWrapper(
        env,
        clip_reward=clip_reward,
        noop_max=noop_max,
        terminal_on_life_loss=episodic_life,
    )
    env = ImageToPyTorch(env)
    if stack_frames > 1:
        env = BufferWrapper(env, stack_frames)
    return env
