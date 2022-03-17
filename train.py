import torch
import torch.nn as nn
import torch.optim as optim
from game.board import BatchBoard
from model.model import DQN, smallDQN, conv_DQN, channel_DQN, full_channel_DQN, full_channel_DQN_v2, ConvNet
from model.greedy_model import GreedyModel
import random
import math
import os, datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from validation import mirror_score
from model.model import NNPlayer

savefreq = 1000
validation_interval = 500
max_score1 = 0
max_score2 = 0
model = ConvNet
greedy_player = GreedyModel()

if __name__ == "__main__":
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BATCH_SIZE = 128
    GAMMA = 0.9
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100_000
    TARGET_UPDATE = 10

    rows = 6
    cols = 7
    batch = 512
    print("BATCH SIZE:", batch)
    policy_net1 = model(rows, cols).to(device)
    target_net1 = model(rows, cols).to(device)
    policy_net2 = model(rows, cols).to(device)
    target_net2 = model(rows, cols).to(device)
    target_net1.load_state_dict(policy_net1.state_dict())
    target_net1.eval()
    target_net2.load_state_dict(policy_net2.state_dict())
    target_net2.eval()

    #summary(policy_net1)
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer1 = optim.Adam(policy_net1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(policy_net2.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    # Create directories for logs
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/fit/" + now_str
    summary_dir = log_dir + "/summary"
    models_dir = log_dir + "/models"
    test_dir = log_dir + "/test"
    os.makedirs(log_dir, exist_ok=True)
    # os.mkdir(summary_dir)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    writer = SummaryWriter(summary_dir)
    interval_tensorboard = 50
    # variable to store the ratio of board filled with coins (this values need to increase)
    mean_ratio_board = torch.zeros([1], device=device)
    # variable to store the mean number of invalid moves (this value need to reduce)
    mean_error_game = torch.zeros([1], device=device)
    mean_loss = 0

    board = BatchBoard(nbatch=batch, device=device)

    list_S = []
    list_F = []
    list_R = []
    list_M = []

    for i in range(10_000_000):
        policy_net1.train()
        policy_net2.train()
        # the policy calculate Q associated at each possible move
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i / EPS_DECAY)
        S = board.state
        with torch.no_grad():
            if board.player == 1:
                Q = target_net1.forward(S).detach()
            else:
                Q = target_net2.forward(S).detach()
        pi = Q.max(dim=1).values
        # take the best move
        M = Q.argmax(dim=1)
        # Sometime choose a random move, using eps_threshold as threshold
        rand_M = torch.randint(0, cols, [batch], device=device)
        # Choose a move using a greedy algorithm
        greedy_M = greedy_player.play(board)
        # When choosing a random move (or a greedy move)
        rand_choice = torch.rand([batch], device=device)
        # Add more randomness at the beginning of the game
        start_random = 0.9
        end_random = 0.5
        decay_random = 10
        randomness_multiplier = end_random + (start_random - end_random) * torch.exp(-1. * board.n_moves / decay_random)
        threshold_multiplied = 1 - (1 - eps_threshold) * (1 - randomness_multiplier)
        # where_play_random = rand_choice < threshold_multiplied
        ratio_greedy = 0.8
        where_play_random = (eps_threshold * ratio_greedy < rand_choice) & (rand_choice < eps_threshold)
        where_play_greedy = rand_choice <= eps_threshold * ratio_greedy
        M[where_play_random] = rand_M[where_play_random]
        M[where_play_greedy] = greedy_M[where_play_greedy]

        list_M.append(M)
        F, R, R_adv, _, _ = board.get_reward(M)

        if len(list_S) >= 2:
            # update rewards and final state using data from opponent play
            list_R[1] += R_adv
            list_F[1][F] = True

            F_old = list_F.pop(0)
            S_old = list_S.pop(0)
            M_old = list_M.pop(0)
            R_old = list_R.pop(0)

            pi[F_old] = 0

            if board.player == -1:
                optimizer1.zero_grad()
                Q_old = policy_net1.forward(S_old)
            else:
                optimizer2.zero_grad()
                Q_old = policy_net2.forward(S_old)

            # PAY ATTENTION to gather method, it is a bit tricky !
            state_action_values = Q_old.gather(1, M_old.unsqueeze(1))
            #Bellman formula
            expected_state_action_values = R_old + GAMMA * pi
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            loss = criterion(state_action_values, expected_state_action_values)
            # loss = torch.abs(delta).sum() / (10 * batch)

            loss.backward()

            if board.player == -1:
                #for param in policy_net1.parameters():
                 #   param.grad.data.clamp_(-1, 1)
                optimizer1.step()
            else:
                #for param in policy_net2.parameters():
                #    param.grad.data.clamp_(-1, 1)
                optimizer2.step()

            #if i % TARGET_UPDATE == 0:

                #target_net1.load_state_dict(policy_net1.state_dict())
                #target_net2.load_state_dict(policy_net2.state_dict())

            # VALIDATION

            if i % validation_interval == 0:
            # Quale modello volete in eval? policynet
                for j, policy, target, max_score in zip([1,2], [policy_net1, policy_net2], [target_net1, target_net2], [max_score1, max_score2]):
                    policy.eval()
                    player = NNPlayer(policy)
                    summary = mirror_score(player, nbatch=batch, n_iter=200, device=device)
                    print(i, j, summary)
                    for key, value in summary.items():
                        writer.add_scalar(f"{key}_{j}", value, i)
                    if summary["score"] > max_score:
                        target.load_state_dict(policy.state_dict())
                    max_score = max(summary["score"], max_score)

            # print(board.n_moves)
            # mean_ratio_board += (board.n_moves.float().mean() / 42.0)
            # mean_error_game += (1.0 * (R==-2)).mean()
            # mean_loss += loss.item()
            # if i % interval_tensorboard == 0:
            #     print(i)
            #     print("Q_old:", Q_old)
            #     print(torch.flip(board.state[0:2], dims=(1,)))
            #     print("M_old:", M_old)
            #     print("R_old:", R_old)
            #     print("F:", F_old)
            #     print("PI:", pi)
            #     print('est:', state_action_values)
            #     print("Mean Ratio Board: ", mean_ratio_board.item() / interval_tensorboard)
            #     print(" Mean error ratio : ", mean_error_game.item() / interval_tensorboard)
            #     print("Current Loss: ", loss)
            #
            #     # TENSOR BOARD
            #     writer.add_scalar("mean_ratio_board",
            #                       mean_ratio_board.item() / interval_tensorboard,
            #                       i)
            #     writer.add_scalar("mean_error_game",
            #                       mean_error_game.item() / interval_tensorboard,
            #                       i)
            #     writer.add_scalar("loss",
            #                       mean_loss / interval_tensorboard,
            #                       i)
            #     mean_ratio_board *= 0
            #     mean_error_game *= 0
            #     mean_loss = 0
            # if i % 10_000 > 9_900:
            #         print(board.state[0])
            if i % savefreq == 0:
                path1 = f"{models_dir}/model_{i}.pth"
                path2 = f"{models_dir}/model-adv_{i}.pth"
                torch.save(policy_net1.state_dict(), path1)
                torch.save(policy_net1.state_dict(), path2)
            # if i % 1_000 == 0:
            #     torch.cuda.empty_cache()

        list_S.append(S)
        list_R.append(R)
        list_F.append(F)


#type "tensorboard --logdir=runs" in terminal