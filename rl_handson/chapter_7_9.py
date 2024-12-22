import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import ale_py
import fire
import gymnasium
import gymnasium as gym
import torch
import torch.multiprocessing as mp
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from lib import calc_loss_dqn, get_device, wrap_dqn
from models import DQN
from ptan import (
    DQNAgent,
    EndOfEpisodeHandler,
    EpisodeEvents,
    EpisodeFPSHandler,
    EpsilonGreedyActionSelector,
    EpsilonTracker,
    ExperienceReplayBuffer,
    ExperienceSourceFirstLast,
    PeriodEvents,
    PeriodicEvents,
    TargetNet,
)
from tensorboardX import SummaryWriter
from torch.optim import Adam

gymnasium.register_envs(ale_py)
from ignite.engine import Engine
from ignite.metrics import RunningAverage

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


def play_func(params: Hyperparams, net: DQN, device: str, exp_queue: mp.Queue):
    env = gym.make(params.env_name)
    env = wrap_dqn(env)
    device = torch.device(device)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = EpsilonTracker(
        selector, params.epsilon_start, params.epsilon_final, params.epsilon_frames
    )
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
        fps_handler: EpisodeFPSHandler,
        initial: int,
        batch_size: int,
    ):
        self.buffer = ExperienceReplayBuffer(
            experience_source=None, buffer_size=buffer_size
        )
        self.exp_queue = exp_queue
        self.fps_handler = fps_handler
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
                    self.fps_handler.step()
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size)


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


def main():
    mp.set_start_method("spawn")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device_str = get_device()
    device = torch.device(device_str)

    params = Hyperparams(
        env_name=DEFAULT_ENV_NAME,
        stop_reward=18.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=1000,
        epsilon_frames=100_000,
    )

    env = gym.make(params.env_name)
    env = wrap_dqn(env)

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = TargetNet(net)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)

    # start subprocess and experience queue
    exp_queue = mp.Queue(maxsize=2)
    proc_args = (params, net, device_str, exp_queue)
    play_proc = mp.Process(target=play_func, args=proc_args)
    play_proc.start()
    fps_handler = EpisodeFPSHandler()
    batch_generator = BatchGenerator(
        params.replay_size,
        exp_queue,
        fps_handler,
        params.replay_initial,
        params.batch_size,
    )

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(
            batch, net, tgt_net.target_model, gamma=params.gamma, device=device
        )
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": batch_generator.epsilon,
        }

    engine = Engine(process_batch)
    EndOfEpisodeHandler(batch_generator, bound_avg_reward=18.0).attach(engine)
    EpisodeFPSHandler().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print(
            "Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s"
            % (
                trainer.state.episode,
                trainer.state.episode_reward,
                trainer.state.episode_steps,
                trainer.state.metrics.get("avg_fps", 0),
                timedelta(seconds=trainer.state.metrics.get("time_passed", 0)),
            )
        )
        trainer.should_terminate = trainer.state.episode > 700

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print(
            "Game solved in %s, after %d episodes and %d iterations!"
            % (
                timedelta(seconds=trainer.state.metrics["time_passed"]),
                trainer.state.episode,
                trainer.state.iteration,
            )
        )
        trainer.should_terminate = True

    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{params.run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v["loss"]).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=["reward", "steps", "avg_reward"]
    )
    tb.attach(
        engine, log_handler=episode_handler, event_name=EpisodeEvents.EPISODE_COMPLETED
    )

    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=["avg_loss", "avg_fps"], output_transform=lambda a: a
    )
    tb.attach(engine, log_handler=handler, event_name=PeriodEvents.ITERS_100_COMPLETED)

    try:
        engine.run(batch_generator)
    finally:
        play_proc.kill()
        play_proc.join()


if __name__ == "__main__":
    fire.Fire(main)
