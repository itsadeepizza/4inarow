import torch
import torch.nn as nn
import torch.optim as optim
from game.game import Board
from model.model import DQN
import random
from collections import namedtuple
import math



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
    #BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100_000
    TARGET_UPDATE = 10

    rows = 6
    cols = 7
    policy_net = DQN(rows, cols).to(device)
    optimizer = optim.RMSprop(policy_net.parameters())
    criterion = nn.SmoothL1Loss()

    board = Board()
    old_Q_move = None
    old_adv_Q_move = None

    for i in range(2_000_000):

        # the policy calculate Q associated at each possible move
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i / EPS_DECAY)
        state = board.state.to(device).unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()
        new_Q = policy_net.forward(state)
        #Best move according to the policy net
        move = torch.argmax(new_Q)
        # Sometime choose a random move
        if random.random() < eps_threshold:
            move = random.randint(0, cols - 1)

        # Stato N - prima della mossa n
        # Calcoliamo Qn
        # Giochiamo la mossa n - reward n
        # l'avversario gioca
        # stato N+1
        # Calcoliamo Q_N+1
        if old_Q_move is not None:
            best_move = torch.argmax(new_Q).detach()
            best_Q = new_Q[0][best_move]
            delta = old_Q_move - (best_Q * GAMMA) - reward
            loss = criterion(delta, torch.zeros(1).to(device))


            # Optimize the model

            loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()


        old_Q_move = new_Q[0][move].detach()
        reward, new_game = board.get_reward(move)
        if i % 100 == 0:
            print(i)
        if i % 10_000 > 9900:
            print("-" * 30)
            print(i)
            print(move)
            print(reward)
            print(board)
            print(old_Q_move, best_Q, reward, delta)
        if i % 50_000 == 0:
            path = f"runs/model_{i//50_000}.pt"
            #torch.save(policy_net.state_dict(), path)
            torch.save(policy_net, path)

        if new_game:
            pass
            #old_Q_move = None
        # The opponent play

        # with torch.no_grad():
        #     Q = policy_net.forward(adv_state)
        #     adv_move = torch.argmax(Q).detach()
        #     # Best move according to the policy net
        #     adv_reward, adv_new_game = board.get_reward(adv_move)

        adv_state = board.state.to(device).unsqueeze(0).unsqueeze(0)
        new_adv_Q = policy_net.forward(state)
        # Best move according to the policy net
        adv_move = torch.argmax(new_adv_Q)
        # Sometime choose a random move
        if random.random() < eps_threshold:
            adv_move = random.randint(0, cols - 1)

        # Stato N - prima della mossa n
        # Calcoliamo Qn
        # Giochiamo la mossa n - reward n
        # l'avversario gioca
        # stato N+1
        # Calcoliamo Q_N+1
        if old_adv_Q_move is not None:
            best_move = torch.argmax(new_adv_Q).detach()
            best_Q = new_adv_Q[0][best_move]
            delta = old_adv_Q_move - (best_Q * GAMMA) - adv_reward
            adv_loss = criterion(delta, torch.zeros(1).to(device))

            # Optimize the model

            adv_loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()

        old_adv_Q_move = new_Q[0][adv_move].detach()
        adv_reward, new_game = board.get_reward(adv_move)





