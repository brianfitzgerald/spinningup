from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.multiprocessing as mp
from models import DQN
from typing import List, Tuple
import random

from ptan import DQNAgent, EpsilonGreedyActionSelector, ExperienceReplayBuffer, ExperienceSourceFirstLast
from lib import calc_loss_dqn, wrap_dqn
from torch.optim import Adam

SEED = 123


@dataclass
class EpisodeEnded:
    reward: float
    steps: int
    epsilon: float


@dataclass
class Hyperparams:
    env_name: str
    stop_reward: float
    run_name: str
    replay_size: int
    replay_initial: int
    target_net_sync: int
    epsilon_frames: int

    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02


class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: Hyperparams):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def play_func(params: Hyperparams, net: DQN, dev_name: str, exp_queue: mp.Queue):
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    device = torch.device(dev_name)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = EpsilonTracker(selector, params)
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=SEED
    )

    for frame_idx, exp in enumerate(exp_source):
        epsilon_tracker.frame(frame_idx // 2)
        exp_queue.put(exp)
        for reward, steps in exp_source.pop_rewards_steps():
            ee = EpisodeEnded(reward=reward, steps=steps, epsilon=selector.epsilon)
            exp_queue.put(ee)


class BatchGenerator:
    def __init__(
        self,
        buffer_size: int,
        exp_queue: mp.Queue,
        initial: int,
        batch_size: int,
    ):
        self.buffer = ExperienceReplayBuffer(
            experience_source=None, buffer_size=buffer_size
        )
        self.exp_queue = exp_queue
        self.initial = initial
        self.batch_size = batch_size
        self._rewards_steps = []
        self.epsilon = None

    def pop_rewards_steps(self) -> List[Tuple[float, int]]:
        res = list(self._rewards_steps)
        self._rewards_steps.clear()
        return res

    def __iter__(self):
        while True:
            while self.exp_queue.qsize() > 0:
                exp = self.exp_queue.get()
                if isinstance(exp, EpisodeEnded):
                    self._rewards_steps.append((exp.reward, exp.steps))
                    self.epsilon = exp.epsilon
                else:
                    self.buffer._add(exp)
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size)


def main():
    mp.set_start_method('spawn')

    random.seed(SEED)
    torch.manual_seed(SEED)
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = gym.make(params.env_name)
    env = wrap_dqn(env)

    net = DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # start subprocess and experience queue
    exp_queue = mp.Queue(maxsize=2)
    proc_args = (params, net, args.dev, exp_queue)
    play_proc = mp.Process(target=play_func, args=proc_args)
    play_proc.start()
    batch_generator = BatchGenerator(
        params.replay_size, exp_queue,
        params.replay_initial, params.batch_size)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": batch_generator.epsilon,
        }

    # TODO replace ignite / engine stuff

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}-{NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)

    try:
        engine.run(batch_generator)
    finally:
        play_proc.kill()
        play_proc.join()
