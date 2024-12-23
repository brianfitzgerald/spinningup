"""
Gym contains an environemnt for playing text-based games.
Each action is a command to be executed in the game, from a preset list

Observations are text sequences, so need to be tokenized and embedded
Env is also a POMDP - since the inventory is not shown, neither is previous state.
So deal with that by stacking states
TextWorld also gives intermediate rewards based on steps to solve the game

"""

import itertools
import typing as tt
from pathlib import Path
from types import SimpleNamespace

import fire
import numpy as np
import ptan
import torch
from gymnasium.spaces import Discrete, Sequence, Space
from ignite.engine import Engine
from lib import get_device
from lib_textworld import TextWorldPreproc, batch_generator, calc_loss_dqn, setup_ignite
from models import DQNConvNet
from ptan import (
    DQNAgent,
    EpisodeEvents,
    ExperienceReplayBuffer,
    ExperienceSourceFirstLast,
    PeriodEvents,
    Preprocessor,
)
from textworld import EnvInfos
from textworld.gym import register_games, register_game
from textworld.text_utils import extract_vocab_from_gamefiles
from torch.optim import RMSprop
from textworld import gym

EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
}


PARAMS = {
    "small": SimpleNamespace(
        **{
            "encoder_size": 20,
            "embeddings": 20,
            "replay_size": 10000,
            "replay_initial": 1000,
            "sync_nets": 100,
            "epsilon_steps": 1000,
            "epsilon_final": 0.2,
        }
    ),
    "medium": SimpleNamespace(
        **{
            "encoder_size": 256,
            "embeddings": 128,
            "replay_size": 100000,
            "replay_initial": 10000,
            "sync_nets": 200,
            "epsilon_steps": 10000,
            "epsilon_final": 0.2,
        }
    ),
}


def get_games_spaces(game_files: tt.List[str]) -> tt.Tuple[
    tt.Dict[int, str],
    Space,
    Space,
]:
    """
    Get games vocabulary, action and observation spaces
    :param game_files: game files to wrap
    :return: tuple with dictionary, action and observation spaces
    """
    vocab = extract_vocab_from_gamefiles(game_files)
    vocab_dict = {idx: word for idx, word in enumerate(sorted(vocab))}
    word_space = Discrete(len(vocab))
    action_space = Sequence(word_space)
    observation_space = Sequence(word_space)
    return vocab_dict, action_space, observation_space


def build_rev_vocab(vocab: tt.Dict[int, str]) -> tt.Dict[str, int]:
    """
    Build reverse vocabulary
    :param vocab: forward vocab (int -> word)
    :return: reverse vocabulary (word -> int)
    """
    res = {word: idx for idx, word in vocab.items()}
    assert len(res) == len(vocab)
    return res


GAMMA = 0.9
LEARNING_RATE = 5e-5
BATCH_SIZE = 64


def main(
    game: str = "simple",
    suffixes: int = 20,
    validation: str = "val",
    run_name: str = "run",
):
    device = get_device()

    game_files = ["games/%s%s.ulx" % (game, s) for s in range(1, suffixes + 1)]
    val_game_file = f"games/{game}{validation}.ulx"
    if not all(map(lambda p: Path(p).exists(), game_files)):
        raise RuntimeError(
            f"Some game files from {game_files} " f"not found! Please run make_games.sh"
        )

    vocab, action_space, observation_space = get_games_spaces(
        game_files + [val_game_file]
    )

    vocab_rev = build_rev_vocab(vocab)
    env_id = register_games(
        gamefiles=game_files, request_infos=EnvInfos(**EXTRA_GAME_INFO), name=game
    )
    print(f"Registered env {env_id} for game files {game_files}")
    val_env_id = register_games(
        gamefiles=[val_game_file], request_infos=EnvInfos(**EXTRA_GAME_INFO), name=game
    )
    print(
        f"Game {val_env_id}, with file {val_game_file} " f"will be used for validation"
    )
    env = gym.make(env_id)
    env = TextWorldPreproc(env, vocab_rev)
    v = env.reset()

    val_env = gym.make(val_env_id)
    val_env = TextWorldPreproc(val_env, vocab_rev)

    params = PARAMS[game]

    prep = Preprocessor(
        dict_size=len(vocab),
        emb_size=params.embeddings,
        num_sequences=env.num_fields,
        enc_output_size=params.encoder_size,
    ).to(device)
    tgt_prep = ptan.agent.TargetNet(prep)

    net = DQNConvNet(obs_size=prep.obs_enc_size, cmd_size=prep.cmd_enc_size)
    net = net.to(device)
    tgt_net = ptan.agent.TargetNet(net)

    agent = DQNAgent(net, prep, epsilon=1, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ExperienceReplayBuffer(exp_source, params.replay_size)

    optimizer = RMSprop(
        itertools.chain(net.parameters(), prep.parameters()), lr=LEARNING_RATE, eps=1e-5
    )

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_t = calc_loss_dqn(
            batch, prep, prep, net, tgt_net.target_model, GAMMA, device=device
        )
        loss_t.backward()
        optimizer.step()
        eps = 1 - engine.state.iteration / params.epsilon_steps
        agent.epsilon = max(params.epsilon_final, eps)
        if engine.state.iteration % params.sync_nets == 0:
            tgt_net.sync()
        return {
            "loss": loss_t.item(),
            "epsilon": agent.epsilon,
        }

    engine = Engine(process_batch)
    run_name = f"tr-{params}_{run_name}"
    save_path = Path("saves") / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    setup_ignite(
        engine, exp_source, run_name, extra_metrics=("val_reward", "val_steps")
    )

    @engine.on(PeriodEvents.ITERS_100_COMPLETED)
    def validate(engine):
        reward = 0.0
        steps = 0

        obs, extra = val_env.reset()

        while True:
            obs_t = prep.encode_observations([obs]).to(device)
            cmd_t = prep.encode_commands(obs["admissible_commands"]).to(device)
            q_vals = net.q_values(obs_t, cmd_t)
            act = np.argmax(q_vals)

            obs, r, is_done, _, _ = val_env.step(act)
            steps += 1
            reward += r
            if is_done:
                break
        engine.state.metrics["val_reward"] = reward
        engine.state.metrics["val_steps"] = steps
        print("Validation got %.3f reward in %d steps" % (reward, steps))
        best_val_reward = getattr(engine.state, "best_val_reward", None)
        if best_val_reward is None:
            engine.state.best_val_reward = reward
        elif best_val_reward < reward:
            print(
                "Best validation reward updated: %s -> %s" % (best_val_reward, reward)
            )
            save_net_name = save_path / ("best_val_%.3f_n.dat" % reward)
            torch.save(net.state_dict(), save_net_name)
            engine.state.best_val_reward = reward

    @engine.on(EpisodeEvents.BEST_REWARD_REACHED)
    def best_reward_updated(trainer: Engine):
        reward = trainer.state.metrics["avg_reward"]
        if reward > 0:
            save_net_name = save_path / ("best_train_%.3f_n.dat" % reward)
            torch.save(net.state_dict(), save_net_name)
            print(
                "%d: best avg training reward: %.3f, saved"
                % (trainer.state.iteration, reward)
            )

    engine.run(batch_generator(buffer, params.replay_initial, BATCH_SIZE))


if __name__ == "__main__":
    fire.Fire(main)
