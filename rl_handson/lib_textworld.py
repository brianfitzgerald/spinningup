import random
import re
import typing as tt
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from gymnasium import spaces
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.envs.registration import EnvSpec
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ptan import (
    BaseAgent,
    EndOfEpisodeHandler,
    EpisodeEvents,
    EpisodeFPSHandler,
    ExperienceFirstLast,
    ExperienceReplayBuffer,
    PeriodEvents,
    PeriodicEvents,
)


def tokenize(text: str, rev_vocab: Dict[str, int]) -> List[int]:
    """
    Very simple tokeniser into fixed word set
    :param text: text to tokenize
    :param rev_vocab: reverse vocabulary
    :return: list of tokens
    """
    res = []
    for word in re.split(r"\W+", text.lower()):
        token = rev_vocab.get(word)
        if token is not None:
            res.append(token)
    return res


KEY_ADM_COMMANDS = "admissible_commands"


class TextWorldPreproc(gym.Wrapper):
    """
    Simple wrapper to preprocess text_world game observation

    Observation and action spaces are not handled, as it will
    be wrapped into other preprocessors
    """

    # field with observation
    OBS_FIELD = "obs"

    def __init__(
        self,
        env: gym.Env,
        vocab_rev: Optional[Dict[str, int]],
        encode_raw_text: bool = False,
        encode_extra_fields: Iterable[str] = ("description", "inventory"),
        copy_extra_fields: Iterable[str] = (),
        use_admissible_commands: bool = True,
        keep_admissible_commands: bool = False,
        use_intermediate_reward: bool = True,
        tokens_limit: Optional[int] = None,
        reward_wrong_last_command: Optional[float] = None,
    ):
        """
        :param env: TextWorld env to be wrapped
        :param vocab_ver: reverse vocabulary
        :param encode_raw_text: flag to encode raw texts
        :param encode_extra_fields: fields to be encoded
        :param copy_extra_fields: fields to be copied into obs
        :param use_admissible_commands: use list of commands
        :param keep_admissible_commands: keep list of admissible commands in observations
        :param use_intermediate_reward: intermediate reward
        :param tokens_limit: limit tokens in encoded fields
        :param reward_wrong_last_command: if given, this reward will be given if 'last_command' observation field is 'None'.
        """
        # Don't call super constructor, to skip the assert
        self.env = env

        self._action_space: spaces.Space[WrapperActType] | None = None
        self._observation_space: spaces.Space[WrapperObsType] | None = None
        self._metadata: dict[str, Any] | None = None

        self._cached_spec: EnvSpec | None = None
        self._vocab_rev = vocab_rev
        self._encode_raw_text = encode_raw_text
        self._encode_extra_field = tuple(encode_extra_fields)
        self._copy_extra_fields = tuple(copy_extra_fields)
        self._use_admissible_commands = use_admissible_commands
        self._keep_admissible_commands = keep_admissible_commands
        self._use_intermediate_reward = use_intermediate_reward
        self._num_fields = len(self._encode_extra_field) + int(self._encode_raw_text)
        self._last_admissible_commands = None
        self._last_extra_info = None
        self._tokens_limit = tokens_limit
        self._reward_wrong_last_command = reward_wrong_last_command
        self._cmd_hist = []

    @property
    def num_fields(self):
        return self._num_fields

    def _maybe_tokenize(self, s: str) -> str | List[int]:
        """
        If dictionary is present, tokenise the string, otherwise keep intact
        :param s: string to process
        :return: tokenized string or original value
        """
        if self._vocab_rev is None:
            return s
        tokens = tokenize(s, self._vocab_rev)
        if self._tokens_limit is not None:
            tokens = tokens[: self._tokens_limit]
        return tokens

    def _encode(self, obs: str, extra_info: dict) -> dict:
        obs_result = []
        if self._encode_raw_text:
            obs_result.append(self._maybe_tokenize(obs))
        for field in self._encode_extra_field:
            extra = extra_info[field]
            obs_result.append(self._maybe_tokenize(extra))
        result = {self.OBS_FIELD: obs_result}
        if self._use_admissible_commands:
            result[KEY_ADM_COMMANDS] = [
                self._maybe_tokenize(cmd) for cmd in extra_info[KEY_ADM_COMMANDS]
            ]
            self._last_admissible_commands = extra_info[KEY_ADM_COMMANDS]
        if self._keep_admissible_commands:
            result[KEY_ADM_COMMANDS] = extra_info[KEY_ADM_COMMANDS]
            if "policy_commands" in extra_info:
                result["policy_commands"] = extra_info["policy_commands"]
        self._last_extra_info = extra_info
        for field in self._copy_extra_fields:
            if field in extra_info:
                result[field] = extra_info[field]
        return result

    def reset(self, seed: Optional[int] = None):
        res, extra = self.env.reset()
        self._cmd_hist = []
        return self._encode(res, extra), extra

    def step(self, action):
        if self._use_admissible_commands:
            action = self._last_admissible_commands[action]
            self._cmd_hist.append(action)
        obs, r, is_done, extra = self.env.step(action)
        if self._use_intermediate_reward:
            r += extra.get("intermediate_reward", 0)
        if self._reward_wrong_last_command is not None:
            if action not in self._last_extra_info[KEY_ADM_COMMANDS]:
                r += self._reward_wrong_last_command
        return self._encode(obs, extra), r, is_done, False, extra

    @property
    def last_admissible_commands(self):
        if self._last_admissible_commands:
            return tuple(self._last_admissible_commands)
        return None

    @property
    def last_extra_info(self):
        return self._last_extra_info


def setup_ignite(
    engine: Engine, exp_source, run_name: str, extra_metrics: tt.Iterable[str] = ()
):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = EndOfEpisodeHandler(exp_source)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    PeriodicEvents().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get("time_passed", 0)
        avg_steps = trainer.state.metrics.get("avg_steps", 50)
        avg_reward = trainer.state.metrics.get("avg_reward", 0.0)
        print(
            "Episode %d: reward=%.0f (avg %.2f), "
            "steps=%s (avg %.2f), speed=%.1f f/s, "
            "elapsed=%s"
            % (
                trainer.state.episode,
                trainer.state.episode_reward,
                avg_reward,
                trainer.state.episode_steps,
                avg_steps,
                trainer.state.metrics.get("avg_fps", 0),
                timedelta(seconds=int(passed)),
            )
        )

        if avg_steps < 15 and trainer.state.episode > 100:
            print("Average steps has fallen below 15, stop training")
            trainer.should_terminate = True

    now = datetime.now().isoformat(timespec="minutes")
    logdir = f"runs/{now}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v["loss"])
    run_avg.attach(engine, "avg_loss")

    metrics = ["reward", "steps", "avg_reward", "avg_steps"]
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    metrics = ["avg_loss", "avg_fps"]
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics, output_transform=lambda a: a
    )
    event = PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)


class Encoder(nn.Module):
    """
    Takes input sequences (after embeddings) and returns
    the hidden state from LSTM
    """

    def __init__(self, emb_size: int, out_size: int):
        super(Encoder, self).__init__()
        self.net = nn.LSTM(input_size=emb_size, hidden_size=out_size, batch_first=True)

    def forward(self, x):
        self.net.flatten_parameters()
        _, hid_cell = self.net(x)
        # Warn: if bidir=True or several layers,
        # sequeeze has to be changed!
        return hid_cell[0].squeeze(0)


class Preprocessor(nn.Module):
    """
    Takes batch of several input sequences and outputs their
    summary from one or many encoders
    """

    def __init__(
        self,
        dict_size: int,
        emb_size: int,
        num_sequences: int,
        enc_output_size: int,
        extra_flags: tt.Sequence[str] = (),
    ):
        """
        :param dict_size: amount of words is our vocabulary
        :param emb_size: dimensionality of embeddings
        :param num_sequences: count of sequences
        :param enc_output_size: output from single encoder
        :param extra_flags: list of fields from observations
        to encode as numbers
        """
        super(Preprocessor, self).__init__()
        self._extra_flags = extra_flags
        self._enc_output_size = enc_output_size
        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoders = []
        for idx in range(num_sequences):
            enc = Encoder(emb_size, enc_output_size)
            self.encoders.append(enc)
            self.add_module(f"enc_{idx}", enc)
        self.enc_commands = Encoder(emb_size, enc_output_size)

    @property
    def obs_enc_size(self):
        return self._enc_output_size * len(self.encoders) + len(self._extra_flags)

    @property
    def cmd_enc_size(self):
        return self._enc_output_size

    def _apply_encoder(self, batch: tt.List[tt.List[int]], encoder: Encoder):
        dev = self.emb.weight.device
        batch_t = [self.emb(torch.tensor(sample).to(dev)) for sample in batch]
        batch_seq = rnn_utils.pack_sequence(batch_t, enforce_sorted=False)
        return encoder(batch_seq)

    def encode_observations(self, observations: tt.List[dict]) -> torch.Tensor:
        sequences = [obs[TextWorldPreproc.OBS_FIELD] for obs in observations]
        res_t = self.encode_sequences(sequences)
        if not self._extra_flags:
            return res_t
        extra = [[obs[field] for field in self._extra_flags] for obs in observations]
        extra_t = torch.Tensor(extra).to(res_t.device)
        res_t = torch.cat([res_t, extra_t], dim=1)
        return res_t

    def encode_sequences(self, batches):
        """
        Forward pass of Preprocessor
        :param batches: list of tuples with variable-length sequences of word ids
        :return: tensor with concatenated encoder outputs for every batch sample
        """
        data = []
        for enc, enc_batch in zip(self.encoders, zip(*batches)):
            data.append(self._apply_encoder(enc_batch, enc))
        res_t = torch.cat(data, dim=1)
        return res_t

    def encode_commands(self, batch):
        """
        Apply encoder to list of commands sequence
        :param batch: list of lists of idx
        :return: tensor with encoded commands in original order
        """
        return self._apply_encoder(batch, self.enc_commands)


@torch.no_grad()
def unpack_batch(
    batch: List[ExperienceFirstLast],
    preprocessor: Preprocessor,
    net: nn.Module,
    device="cpu",
):
    """
    Convert batch to data needed for Bellman step
    :param batch: list of ptan.Experience objects
    :param preprocessor: emb.Preprocessor instance
    :param net: network to be used for next state approximation
    :param device: torch device
    :return: tuple (list of states, list of taken commands,
                    list of rewards, list of best Qs for the next s)
    """
    # calculate Qs for next states
    states, taken_commands, rewards, best_q = [], [], [], []
    last_states, last_commands, last_offsets = [], [], []
    for exp in batch:
        states.append(exp.state)
        taken_commands.append(exp.state["admissible_commands"][exp.action])
        rewards.append(exp.reward)

        # calculate best Q value for the next state
        if exp.last_state is None:
            # final state in the episode, Q=0
            last_offsets.append(len(last_commands))
        else:
            assert isinstance(exp.last_state, dict)
            last_states.append(exp.last_state)
            last_commands.extend(exp.last_state["admissible_commands"])
            last_offsets.append(len(last_commands))

    obs_t = preprocessor.encode_observations(last_states).to(device)
    commands_t = preprocessor.encode_commands(last_commands).to(device)

    prev_ofs = 0
    obs_ofs = 0
    for ofs in last_offsets:
        if prev_ofs == ofs:
            best_q.append(0.0)
        else:
            q_vals = net.q_values(
                obs_t[obs_ofs : obs_ofs + 1], commands_t[prev_ofs:ofs]
            )
            best_q.append(max(q_vals))
            obs_ofs += 1
        prev_ofs = ofs
    return states, taken_commands, rewards, best_q


def batch_generator(buffer: ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def calc_loss_dqn(
    batch, preprocessor, tgt_preprocessor, net, tgt_net, gamma, device="cpu"
):
    states, taken_commands, rewards, next_best_qs = unpack_batch(
        batch, tgt_preprocessor, tgt_net, device
    )

    obs_t = preprocessor.encode_observations(states).to(device)
    cmds_t = preprocessor.encode_commands(taken_commands).to(device)
    q_values_t = net(obs_t, cmds_t)
    tgt_q_t = torch.tensor(rewards) + gamma * torch.tensor(next_best_qs)
    tgt_q_t = tgt_q_t.to(device)
    return F.mse_loss(q_values_t.squeeze(-1), tgt_q_t)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        net: nn.Module,
        preprocessor: Preprocessor,
        epsilon: float = 0.0,
        device="cpu",
    ):
        self.net = net
        self._prepr = preprocessor
        self._epsilon = epsilon
        self.device = device

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if 0.0 <= value <= 1.0:
            self._epsilon = value

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        # for every state in the batch, calculate
        actions = []
        for state in states:
            commands = state["admissible_commands"]
            if random.random() <= self.epsilon:
                actions.append(random.randrange(len(commands)))
            else:
                obs_t = self._prepr.encode_observations([state]).to(self.device)
                commands_t = self._prepr.encode_commands(commands)
                commands_t = commands_t.to(self.device)
                q_vals = self.net.q_values(obs_t, commands_t)
                actions.append(np.argmax(q_vals))
        return actions, agent_states
