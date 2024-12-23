import re
import typing as tt
import warnings
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import gymnasium as gym
import torch.utils.tensorboard as tb_logger
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ptan import (
    EndOfEpisodeHandler,
    EpisodeEvents,
    EpisodeFPSHandler,
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
        super(TextWorldPreproc, self).__init__(env)
        self._vocab_rev = vocab_rev
        self._encode_raw_text = encode_raw_text
        self._encode_extra_field = tuple(encode_extra_fields)
        self._copy_extra_fields = tuple(copy_extra_fields)
        self._use_admissible_commands = use_admissible_commands
        self._keep_admissible_commands = keep_admissible_commands
        self._use_intermedate_reward = use_intermediate_reward
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
    tb = tb_logger.SummaryWriter(log_dir=logdir)
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
