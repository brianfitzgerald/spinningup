"""
AlphaGo is a model-based method, in contrast with AlphaZero, which is model-free.
PPO and DQN are also model-free methods, since they learn the policy or the value function directly from the environment,
as opposed to learning a model of the environment and then using it to plan.

Model based methods are more sample efficient, and can be more data efficient, but they are also more complex and harder to train.
Model refers to a model of the environment
Once a model is learned, it can be used to plan, which can be more efficient than learning the policy or value function directly.
Then, we have less of a dependency on the real environment, and can use the model to generate more data.
Models are also transferable - we can learn a model in one environment and use it in another, or for different tasks.

AlphaGo overview:
- MCTS algorithm for traversing game states
    - core idea is to randomly walk down the game states, expand them and gather statistics about the states
    - then, use these statistics to guide the search towards the most promising states
- Use a neural network to evaluate game states during the MCTS search
- then, use self play to train the neural network, by playing games against itself

MCTS:
- branching factor is the number of possible actions at any state in the game
- Total number of possible game states is the branching factor raised to the power of the depth of the game tree
- To deal with the huge number of possible states, MCTS uses a tree search algorithm
- Depth-first search - starting at the current game state, select the most promising action, or a random action, then repeat the process from the new state
- Process is similar to value iteration, where episodes are playedr and the final step of the episode is used to update the value function

In AlphaGo:
- For every edge, store:
    - Prior probability of the edge
    - Visit count
    - Total action value
- Select an action following Q(s,a) + U(s,a) formula, where U(s,a) is the prior probability of the edge
- This is calculated as P(s, a) / 1 + N(s, a), where P(s, a) is the prior probability and N(s, a) is the visit count
- Randomness is added to the selection process by adding noise to the prior probabilities
- A neural network is used to obtain the prior probabilities, and the value of the state estimation, Q(s)
- Once the value is obtained by ending the game, or continuing the search, the backup operation is performed - the value is propagated back up the tree, though each visited intermediate node
    - We update the visit count, N(s, a), and the total action value, Q(s, a)
- This search process is performed 1-2k times, each move
- Self-play is used to train the neural network
    - Start with completely random moves, and transition to deterministic moves as the network is trained.
    - Select the action with the largest visit count
- Once self-play is finished, each step of the game is added to the training dataset
- Use minibatches from the replay buffer, filled from the self-play games, to train the neural network
- Minimize the MSE between the value head position and the actual position value, as well as the CE loss between the policy head and the actual policy

"""

import os
import random
import time

import fire
import torch
import torch.optim as optim
from lib import get_device
from ptan import (
    TargetNet,
    TBMeanTracker,
)
from torch.utils.tensorboard.writer import SummaryWriter

from muzero import Net

PLAY_EPISODES = 1  # 25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 5000  # 30000
LEARNING_RATE = 0.001
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000  # 10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10


def evaluate(net1: Net, net2: Net, rounds: int, device: torch.device):
    n1_win, n2_win = 0, 0
    mcts_stores = [MCTS(), MCTS()]

    for r_idx in range(rounds):
        r, _ = play_game(
            mcts_stores=mcts_stores,
            replay_buffer=None,
            net1=net1,
            net2=net2,
            steps_before_tau_0=0,
            mcts_searches=20,
            mcts_batch_size=16,
            device=device,
        )
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1
    return n1_win / (n1_win + n2_win)


def main(name: str = "mcts"):
    device = get_device()
    saves_path = os.path.join("saves", name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + name)

    net = Net(input_shape=model.OBS_SHAPE, actions_n=game.GAME_COLS).to(device)
    best_net = TargetNet(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = deque(maxlen=REPLAY_BUFFER)
    mcts_store = mcts.MCTS()
    step_idx = 0
    best_idx = 0

    with TBMeanTracker(writer, batch_size=10) as tb_tracker:
        while True:
            t = time.time()
            prev_nodes = len(mcts_store)
            game_steps = 0
            for _ in range(PLAY_EPISODES):
                _, steps = play_game(
                    mcts_store,
                    replay_buffer,
                    best_net.target_model,
                    best_net.target_model,
                    steps_before_tau_0=STEPS_BEFORE_TAU_0,
                    mcts_searches=MCTS_SEARCHES,
                    mcts_batch_size=MCTS_BATCH_SIZE,
                    device=device,
                )
                game_steps += steps
            game_nodes = len(mcts_store) - prev_nodes
            dt = time.time() - t
            speed_steps = game_steps / dt
            speed_nodes = game_nodes / dt
            tb_tracker.track("speed_steps", speed_steps, step_idx)
            tb_tracker.track("speed_nodes", speed_nodes, step_idx)
            print(
                "Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d"
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
                states_v = model.state_lists_to_batch(
                    batch_states_lists, batch_who_moves, device
                )

                optimizer.zero_grad()
                probs_v = torch.FloatTensor(batch_probs).to(device)
                values_v = torch.FloatTensor(batch_values).to(device)
                out_logits_v, out_values_v = net(states_v)

                loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
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
                    net, best_net.target_model, rounds=EVALUATION_ROUNDS, device=device
                )
                print("Net evaluated, win ratio = %.2f" % win_ratio)
                writer.add_scalar("eval_win_ratio", win_ratio, step_idx)
                if win_ratio > BEST_NET_WIN_RATIO:
                    print("Net is better than cur best, sync")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(
                        saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx)
                    )
                    torch.save(net.state_dict(), file_name)
                    mcts_store.clear()


if __name__ == "__main__":
    fire.Fire(main)
