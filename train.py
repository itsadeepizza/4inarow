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


class Trainer():

    def __init__(self, batch_size, hyperparams, model, rows=6, cols=7):
        # if gpu is to be used
        # SETTING PARAMETERS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in hyperparams:
            setattr(self, key, hyperparams[key])

        self.rows = rows
        self.cols = cols
        self.batch_size = batch_size
        print("BATCH SIZE:", batch_size)
        # INITIALISING MODELS
        self.policy_net1 = model(rows, cols).to(self.device)
        self.target_net1 = model(rows, cols).to(self.device)
        self.policy_net2 = model(rows, cols).to(self.device)
        self.target_net2 = model(rows, cols).to(self.device)
        # SET TARGET MODEL
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net1.eval()
        self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.target_net2.eval()
        # GREEDY MODEL
        self.greedy_player = GreedyModel()
        # OPTIMIZER AND CRITERION
        self.optimizer1 = optim.Adam(self.policy_net1.parameters(), lr=self.lr)
        self.optimizer2 = optim.Adam(self.policy_net2.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        # TENSORBOARD AND LOGGING
        # Create directories for logs
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "runs/fit/" + now_str
        self.summary_dir = self.log_dir + "/summary"
        self.models_dir = self.log_dir + "/models"
        self.test_dir = self.log_dir + "/test"
        os.makedirs(self.log_dir, exist_ok=True)
        # os.mkdir(summary_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        self.writer = SummaryWriter(self.summary_dir)
        # variable to store the ratio of board filled with coins (this values need to increase)
        self.mean_ratio_board = torch.zeros([1], device=self.device)
        # variable to store the mean number of invalid moves (this value need to reduce)
        self.mean_error_game = torch.zeros([1], device=self.device)
        self.mean_loss = 0
        # VARIABLES FOR TRAINING
        self.board = BatchBoard(nbatch=self.batch_size, device=self.device)
        self.list_S = []
        self.list_F = []
        self.list_R = []
        self.list_M = []
        self.max_score1 = 0
        self.max_score2 = 0

    def train(self):
        for i in range(10_000_000):
            self.train_move(i)


    def train_move(self, i):
        self.policy_net1.train()
        self.policy_net2.train()
        # Update threeshold
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i / self.EPS_DECAY)
        # Choose next move
        S = self.board.state
        with torch.no_grad():
            if self.board.player == 1:
                Q = self.target_net1.forward(S).detach()
            else:
                Q = self.target_net2.forward(S).detach()
        pi = Q.max(dim=1).values
        # take the best move
        M = Q.argmax(dim=1)
        # Sometime choose a random move, using eps_threshold as threshold
        rand_M = torch.randint(0, self.cols, [self.batch_size], device=self.device)
        # Choose a move using a greedy algorithm
        greedy_M = self.greedy_player.play(self.board)
        # When choosing a random move (or a greedy move)
        rand_choice = torch.rand([self.batch_size], device=self.device)
        # Add more randomness at the beginning of the game
        randomness_multiplier = self.end_random + (self.start_random - self.end_random) * torch.exp(-1. * self.board.n_moves / self.decay_random)
        threshold_multiplied = 1 - (1 - eps_threshold) * (1 - randomness_multiplier)
        # where_play_random = rand_choice < threshold_multiplied
        where_play_random = (eps_threshold * self.ratio_greedy < rand_choice) & (rand_choice < eps_threshold)
        where_play_greedy = rand_choice <= eps_threshold * self.ratio_greedy
        M[where_play_random] = rand_M[where_play_random]
        M[where_play_greedy] = greedy_M[where_play_greedy]
        # The move is played
        self.list_M.append(M)
        F, R, R_adv, _, _ = self.board.get_reward(M)

        if len(self.list_S) >= 2:
            # update rewards and final state using data from opponent play
            self.list_R[1] += R_adv
            self.list_F[1][F] = True

            F_old = self.list_F.pop(0)
            S_old = self.list_S.pop(0)
            M_old = self.list_M.pop(0)
            R_old = self.list_R.pop(0)

            pi[F_old] = 0

            if self.board.player == -1:
                self.optimizer1.zero_grad()
                Q_old = self.policy_net1.forward(S_old)
            else:
                self.optimizer2.zero_grad()
                Q_old = self.policy_net2.forward(S_old)

            # PAY ATTENTION to gather method, it is a bit tricky !
            state_action_values = Q_old.gather(1, M_old.unsqueeze(1))
            # Bellman formula
            expected_state_action_values = R_old + self.GAMMA * pi
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            loss = self.criterion(state_action_values, expected_state_action_values)
            # loss = torch.abs(delta).sum() / (10 * batch)

            loss.backward()

            if self.board.player == -1:
                # for param in policy_net1.parameters():
                #   param.grad.data.clamp_(-1, 1)
                self.optimizer1.step()
            else:
                # for param in policy_net2.parameters():
                #    param.grad.data.clamp_(-1, 1)
                self.optimizer2.step()

            # if i % TARGET_UPDATE == 0:
                # Now target method is replaced when the policy perform best than current target
                # target_net1.load_state_dict(policy_net1.state_dict())
                # target_net2.load_state_dict(policy_net2.state_dict())

            # VALIDATION

            if i % self.validation_interval == 0:
                # Quale modello volete in eval? policynet
                self.update_target(i)

            # print(board.n_moves)
            # mean_ratio_board += (board.n_moves.float().mean() / 42.0)
            # mean_error_game += (1.0 * (R==-2)).mean()
            self.mean_loss += loss.item()
            if i % self.interval_tensorboard == 0:
                self.report(i)

            # if i % 10_000 > 9_900:
            #         print(board.state[0])
            if i % self.savefreq == 0:
                self.save_model(i)

            # if i % 1_000 == 0:
            #     torch.cuda.empty_cache()

        self.list_S.append(S)
        self.list_R.append(R)
        self.list_F.append(F)

    def save_model(self, i):
        path1 = f"{self.models_dir}/model_{i}.pth"
        path2 = f"{self.models_dir}/model-adv_{i}.pth"
        torch.save(self.policy_net1.state_dict(), path1)
        torch.save(self.policy_net1.state_dict(), path2)

    def report(self, i):
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
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i / self.EPS_DECAY)
        self.writer.add_scalar("eps_threshold",
                          eps_threshold,
                          i)
        self.writer.add_scalar("loss",
                               self.mean_loss / self.interval_tensorboard,
                               i)
        #     mean_ratio_board *= 0
        #     mean_error_game *= 0
        self.mean_loss = 0


    def update_target(self, i):
        for j, policy, target, max_score in zip([1, 2],
                                                [self.policy_net1, self.policy_net2],
                                                [self.target_net1, self.target_net2],
                                                [self.max_score1, self.max_score2]):
            policy.eval()
            player = NNPlayer(policy)
            summary = mirror_score(player, nbatch=self.batch_size, n_iter=200, device=self.device)
            print(i, j, summary)
            for key, value in summary.items():
                self.writer.add_scalar(f"{key}_{j}", value, i)
            if summary["score"] > max_score:
                target.load_state_dict(policy.state_dict())
            max_score = max(summary["score"], max_score)



if __name__ == "__main__":
    hyperparams = {
        "GAMMA": 0.9,
        "EPS_START": 0.9,
        "EPS_END": 0.05,
        "EPS_DECAY": 100_000,
        "TARGET_UPDATE": 10,
        "start_random": 0.9,
        "end_random": 0.5,
        "decay_random": 10,
        "ratio_greedy": 0.8,
        "lr": 0.001,
        "interval_tensorboard": 50,
        "savefreq": 1000,
        "validation_interval": 500
    }
    model = ConvNet
    trainer = Trainer(batch_size=10, hyperparams=hyperparams, model=model)
    trainer.train()


#type "tensorboard --logdir=runs" in terminal