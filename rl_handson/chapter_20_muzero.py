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
import time
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

PLAY_EPISODES = 1  # 25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER_SIZE = 5000  # 30000
LEARNING_RATE = 0.001
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000  # 10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10


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
    net = Net(input_shape=game.obs_shape, actions_n=game.cols, num_filters=16).to(
        device
    )
    best_net = TargetNet(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    assert (
        MIN_REPLAY_TO_TRAIN >= BATCH_SIZE
    ), "Replay buffer size should be larger than batch size"

    # format is (state, cur_player, probs, result)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    mcts_store = MCTS(game)
    step_idx = 0
    best_idx = 0

    with TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            t = time.time()
            prev_nodes = len(mcts_store)
            game_steps = 0
            for _ in range(PLAY_EPISODES):
                # Play a single round of the game - connect 4 or chess
                _, steps = play_game(
                    game=game,
                    mcts_stores=mcts_store,
                    replay_buffer=replay_buffer,
                    net1=net,
                    net2=best_net.target_model,
                    steps_before_tau_0=STEPS_BEFORE_TAU_0,
                    mcts_searches=MCTS_SEARCHES,
                    mcts_batch_size=MCTS_BATCH_SIZE,
                    device=device,
                )
                game_steps += steps

            # Track the speed of the game
            game_nodes = len(mcts_store) - prev_nodes
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt
            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            logger.info(
                "Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay buffer len %d"
                % (
                    step_idx,
                    game_steps,
                    game_nodes,
                    speed_steps,
                    speed_nodes,
                    best_idx,
                    len(replay_buffer),
                )
            )
            step_idx += 1

            if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
                continue

            # train
            sum_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0

            for _ in range(TRAIN_ROUNDS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
                batch_states_lists = [
                    game.decode_binary(state) for state in batch_states
                ]
                # get batch of game states
                states_v = state_lists_to_batch(
                    batch_states_lists, batch_who_moves, game, device
                )

                optimizer.zero_grad()
                probs_v = torch.FloatTensor(batch_probs).to(device)
                values_v = torch.FloatTensor(batch_values).to(device)
                # states_v is (256, 2, 6, 7) - player, then board
                # different channels per player so the game state is invariant to which player is playing
                # value head returns the value of every possible action, and a single value float which is the value of the state
                out_logits_v, out_values_v = net(states_v)

                out_values_v = out_values_v.squeeze(-1)
                loss_value_v = F.mse_loss(out_values_v, values_v)
                loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
                loss_policy_v = loss_policy_v.sum(dim=1).mean()

                loss_v = loss_policy_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                sum_loss += loss_v.item()
                sum_value_loss += loss_value_v.item()
                sum_policy_loss += loss_policy_v.item()

            tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
            tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

            # evaluate net
            if step_idx % EVALUATE_EVERY_STEP == 0:
                win_ratio = evaluate(
                    net,
                    best_net.target_model,
                    rounds=EVALUATION_ROUNDS,
                    game=game,
                    device=device,
                )
                logger.info("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    logger.info("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(
                        saves_path, f"best_{best_idx}_{win_ratio}_%.3f.dat"
                    )
                    torch.save(net.state_dict(), file_name)
                    mcts_store.clear()


if __name__ == "__main__":
    fire.Fire(main)
