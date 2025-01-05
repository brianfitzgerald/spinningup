"""

MuZero:
- perform MCTS but don't rely on game state / model
- Add two extra neural networks - a dynamics model and a representation model
    - the representation model is used to encode the game state into a latent representation
    - the dynamics model is used to predict the next state and the reward
- use the value model to compute the value of an action-state pair, then select the action with the highest value for the MCTS search
- If this is the first time the state is visited, use the dynamics model to predict the next state and the reward
- This process is repeated hundreds of times, then the MCTS search is used to select the best action
    - the process of updating the tree is named backprop, instead of backup in alphazero

MuZero algorithm:
- Select the action from a root state based on the visit count of the edges
- Then the selected action is played, and the tree is updated, and we perform another MCTS search from the new state
- Take generated episodes, and use them to train the neural networks via the replay buffer

Tree uses upper confidence bounds instead of the value of the state
"""

import os
import random
from time import time
from collections import deque

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from game import ConnectFour
from lib import ensure_directory, get_device
from loguru import logger
from mcts import MCTS, state_lists_to_batch
from rl_handson.alphazero import Net, play_game
from ptan import (
    TargetNet,
    TBMeanTracker,
)
from torch.utils.tensorboard.writer import SummaryWriter


# count of recent episodes in the replay buffer
REPLAY_BUFFER = 128

# this amount of states will be generated, samples are *5 of this size
BATCH_SIZE = 256
TRAIN_ROUNDS = 1
LEARNING_RATE = 0.001
TEMP_START = 10
TEMP_STEP = 0.1

BEST_NET_WIN_RATIO = 0.7
EVALUATE_EVERY_STEP = 10
EVALUATION_ROUNDS = 20


def evaluate(
    net1: nn.Module, net2: nn.Module, rounds: int, game: ConnectFour, device: str
):
    n1_win, n2_win = 0, 0
    mcts_stores = [MCTS(game), MCTS(game)]

    for r_idx in range(rounds):
        r, _ = play_game(
            game=game,
            mcts_stores=mcts_stores,
            replay_buffer=None,
            net1=net1,
            net2=net2,
            steps_before_tau_0=0,
            mcts_searches=20,
            mcts_batch_size=16,
            device=device,
        )
        assert r is not None
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1
    return n1_win / (n1_win + n2_win)


def main(name: str = "mcts"):
    device = get_device()
    saves_path = os.path.join("saves", name)
    ensure_directory(saves_path)
    writer = SummaryWriter(comment="-" + name)

    game = ConnectFour()
    # format is (state, cur_player, probs, result)
    replay_buffer = deque(maxlen=REPLAY_BUFFER)
    step_idx = 0
    best_idx = 0

    with TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            step_idx += 1
            ts = time()
            score, episode = play_game(best_net, best_net, params, temperature=temperature)
            print(f"{step_idx}: result {score}, steps={len(episode)}")
            replay_buffer.append(episode)
            writer.add_scalar("time_play", time() - ts, step_idx)
            writer.add_scalar("steps", len(episode), step_idx)

            # training
            ts = time()
            for _ in range(TRAIN_ROUNDS):
                states_t, actions, policy_tgt, rewards_tgt, values_tgt = \
                    sample_batch(replay_buffer, BATCH_SIZE, params)

                optimizer.zero_grad()
                h_t = net.repr(states_t)
                loss_p_full_t = None
                loss_v_full_t = None
                loss_r_full_t = None
                for step in range(params.unroll_steps):
                    policy_t, values_t = net.pred(h_t)
                    loss_p_t = F.cross_entropy(policy_t, policy_tgt[step])
                    loss_v_t = F.mse_loss(values_t, values_tgt[step])
                    # dynamic step
                    rewards_t, h_t = net.dynamics(h_t, actions[step])
                    loss_r_t = F.mse_loss(rewards_t, rewards_tgt[step])
                    if step == 0:
                        loss_p_full_t = loss_p_t
                        loss_v_full_t = loss_v_t
                        loss_r_full_t = loss_r_t
                    else:
                        loss_p_full_t += loss_p_t * 0.5
                        loss_v_full_t += loss_v_t * 0.5
                        loss_r_full_t += loss_r_t * 0.5
                loss_full_t = loss_v_full_t + loss_p_full_t + loss_r_full_t
                loss_full_t.backward()
                optimizer.step()

                writer.add_scalar("loss_p", loss_p_full_t.item(), step_idx)
                writer.add_scalar("loss_v", loss_v_full_t.item(), step_idx)
                writer.add_scalar("loss_r", loss_r_full_t.item(), step_idx)


            writer.add_scalar("time_train", time() - ts, step_idx)
            writer.add_scalar("temp", temperature, step_idx)
            temperature = max(0.0, temperature - TEMP_STEP)

            # evaluate net
            if step_idx % EVALUATE_EVERY_STEP == 0:
                ts = time()
                win_ratio, avg_steps = evaluate(net, best_net, params)
                print("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                writer.add_scalar("eval_steps", avg_steps, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    print("Net is better than cur best, sync")
                    best_net.sync(net)
                    best_idx += 1
                    file_name = os.path.join(saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx))
                    torch.save(best_net.get_state_dict(), file_name)
                writer.add_scalar("time_eval", time() - ts, step_idx)


if __name__ == "__main__":
    fire.Fire(main)
