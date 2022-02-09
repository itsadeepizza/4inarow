import torch
import torch.nn as nn
import torch.optim as optim
from game.game import BatchBoard
from model.model import DQN
import random
import math
import os, datetime
from torch.utils.tensorboard import SummaryWriter
# def optimize_model():
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                                 if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

if __name__ == "__main__":
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100_000
    TARGET_UPDATE = 10

    rows = 6
    cols = 7
    batch = 1_000
    policy_net = DQN(rows, cols).to(device)
    target_net = DQN(rows, cols).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
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
    # variable to store the ratio of case used in the grid
    mean_ratio_board = 0
    # variable to store the mean number of invalid moves
    mean_lost_game = 0
    board = BatchBoard(nbatch=batch, device=device)

    list_S = []
    list_F = []
    list_R = []
    list_M = []

    for i in range(10_000_000):

        # the policy calculate Q associated at each possible move
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i / EPS_DECAY)
        S = board.state
        with torch.no_grad():
            Q = target_net.forward(S).detach()
        pi = Q.max(dim=1).values
        # take the best move
        M = Q.argmax(dim=1)
        # Sometime choose a random move, using eps_threshold as threshold
        rand_M = torch.randint(0, cols, [batch], device=device)
        rand_choice = torch.rand([batch], device=device)
        M[rand_choice < eps_threshold] = rand_M[rand_choice < eps_threshold]

        list_M.append(M)
        F, R, R_adv = board.get_reward(M)

        if len(list_S) >= 2:
            # update rewards and final state using data from opponent play
            list_R[1] += R_adv
            list_F[1][F] = True

            F_old = list_F.pop(0)
            S_old = list_S.pop(0)
            M_old = list_M.pop(0)
            R_old = list_R.pop(0)

            pi[F_old] = 0

            Q_old = policy_net.forward(S_old)
            state_action_values = torch.gather(Q_old, 1, M_old.unsqueeze(0))
            expected_state_action_values = R_old + GAMMA * pi

            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))
            # loss = torch.abs(delta).sum() / (10 * batch)
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            if i % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            mean_ratio_board += torch.abs(board.state).mean()
            mean_lost_game = ((R==-2)/(-2)).mean()
            interval = 1000
            if i % interval == 0:
                print(i)
                print("Q:", Q)
                #print(board.state[0:2])
                #print("R:", R)
                #print("F:", F_old)
                #print("PI:", pi)
                #print('est:', state_action_values)
                print(mean_ratio_board.item())
                print(mean_lost_game.item())
                writer.add_scalar("mean_ratio_board",
                                  mean_ratio_board.item() / interval,
                                  i)
                writer.add_scalar("mean_lost_game",
                                  mean_lost_game.item() / interval,
                                  i)
                mean_ratio_board = 0
                mean_lost_game = 0
            if i % 10_000 > 9_900:
                    pass
                    print(board.state[0])
            if i % 30_000 == 0:
                path = f"{models_dir}/model_{i}.pth"
                torch.save(policy_net.state_dict(), path)
            if i % 1_000 == 0:
                torch.cuda.empty_cache()



        list_S.append(S)
        list_R.append(R)
        list_F.append(F)


#type "tensorboard --logdir=runs" in terminal