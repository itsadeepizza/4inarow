import torch
import torch.nn as nn
import torch.optim as optim
from game.board import BatchBoard
from model.model import DQN, smallDQN, conv_DQN, channel_DQN, full_channel_DQN, full_channel_DQN_v2, ConvNet, ConvNetNoMem
from model.greedy_model import GreedyModel
import math
from validation import mirror_score
from model.model_helper import NNPlayer
from base_trainer import BaseTrainer



class Trainer(BaseTrainer):

    def __init__(self,  batch_size, hyperparams: dict, model, trainer_player, rows=6, cols=7, device=None, seed=None):
        super().__init__(batch_size, hyperparams, model, device=device, seed=seed)
        self.rows = rows
        self.cols = cols
        self.model = model
        self.init_models()
        # GREEDY MODEL
        self.trainer_player = trainer_player
        self.cum_loss1 = 0
        self.cum_loss2 = 0


        # VARIABLES FOR TRAINING
        self.board = BatchBoard(nbatch=self.batch_size, device=self.device)
        self.list_S = []
        self.list_F = []
        self.list_R = []
        self.list_M = []
        self.max_score1 = 0
        self.max_score2 = 0

        # Memory
        try:
            self.model.memory_size
        except:
            # The model do not use memory
            self.use_memory = False
        else:
            # the model use memory
            self.use_memory = True
        if self.use_memory:
            self.memory1 : torch.Tensor = torch.zeros((self.batch_size, *self.model.memory_size), device=self.device)
            self.memory2 : torch.Tensor = self.memory1.clone()


    def init_models(self):
        # INITIALISING MODELS
        self.policy_net1 = self.model(self.rows, self.cols).to(self.device)
        self.target_net1 = self.model(self.rows, self.cols).to(self.device)
        self.policy_net2 = self.model(self.rows, self.cols).to(self.device)
        self.target_net2 = self.model(self.rows, self.cols).to(self.device)
        # SET TARGET MODEL
        self.target_net1.load_state_dict(self.policy_net1.state_dict())
        self.target_net1.eval()
        self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.target_net2.eval()
        # OPTIMIZER AND CRITERION
        self.optimizer1 = optim.Adam(self.policy_net1.parameters(), lr=self.lr)
        self.optimizer2 = optim.Adam(self.policy_net2.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()


    def init_logger(self):
        # TENSORBOARD AND LOGGING
        super().init_logger()
        self.mean_ratio_board = torch.zeros([1], device=self.device)
        # variable to store the mean number of invalid moves (this value need to reduce)
        self.mean_error_game = torch.zeros([1], device=self.device)

    def choose_move(self, i):
        # Update threeshold
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i / self.EPS_DECAY)
        # Choose next move
        S = self.board.state
        with torch.no_grad():
            if self.board.player == 1:
                # Q is the long term reward for each move
                out = self.target_net1.forward(S)
            else:
                out = self.target_net2.forward(S)
        if self.use_memory:
            Q = out[0].detach()
        else:
            Q = out.detach()
        # Long term reward associated to the best move
        pi = Q.max(dim=1).values
        # take the best move
        M = Q.argmax(dim=1)
        # Sometime choose a random move, using eps_threshold as threshold
        rand_M = torch.randint(0, self.cols, [self.batch_size], device=self.device)
        # Choose a move using the trainer algorithm (ex greedy algorithm)
        trainer_M = self.trainer_player.play(self.board)
        # When choosing a random move (or a trainer move)
        rand_choice = torch.rand([self.batch_size], device=self.device)
        # Add more randomness at the beginning of the game
        randomness_multiplier = self.end_random + (self.start_random - self.end_random) * torch.exp(
            -1. * self.board.n_moves / self.decay_random)
        threshold_multiplied = 1 - (1 - eps_threshold) * (1 - randomness_multiplier)
        # where_play_random = rand_choice < threshold_multiplied
        # where_play_random = (eps_threshold * self.ratio_greedy < rand_choice) & (rand_choice < eps_threshold)
        # where_play_greedy = rand_choice <= eps_threshold * self.ratio_trainer
        where_play_random = (threshold_multiplied * self.ratio_trainer < rand_choice) & (
                    rand_choice < threshold_multiplied)
        where_play_greedy = rand_choice <= threshold_multiplied * self.ratio_trainer
        M[where_play_random] = rand_M[where_play_random]
        M[where_play_greedy] = trainer_M[where_play_greedy]
        return M, pi, S

    def train_move(self, i):
        self.policy_net1.train()
        self.policy_net2.train()
        # The move is chosen (target network + greedy + random)
        M, pi, S = self.choose_move(i)
        self.list_M.append(M)
        # The move is played
        summary = self.board.get_reward(M)
        F = summary["is_final"]
        R = summary["rewards"]
        R_adv = summary["adv_rewards"]

        if len(self.list_S) >= 2:
            # update rewards and final state using data from opponent play
            self.list_R[1] += R_adv
            self.list_F[1][F] = True

            F_old = self.list_F.pop(0) # List of which games are final states
            S_old = self.list_S.pop(0) # List of states
            M_old = self.list_M.pop(0) # List of played moves
            R_old = self.list_R.pop(0) # List of rewards
            # The long term reward associated to the initial state is 0
            pi[F_old] = 0

            if self.board.player == -1:
                self.optimizer1.zero_grad()
                if self.use_memory:
                    Q_old, self.memory1 = self.policy_net1.forward(S_old, self.memory1)
                else:
                    Q_old = self.policy_net1.forward(S_old)
            else:
                self.optimizer2.zero_grad()
                if self.use_memory:
                    Q_old, self.memory2 = self.policy_net2.forward(S_old, self.memory2)
                else:
                    Q_old = self.policy_net2.forward(S_old)

            # PAY ATTENTION to gather method, it is a bit tricky !
            state_action_values = Q_old.gather(1, M_old.unsqueeze(1))
            # Bellman formula
            expected_state_action_values = R_old + self.GAMMA * pi
            expected_state_action_values = expected_state_action_values.unsqueeze(1)

            loss = self.criterion(state_action_values, expected_state_action_values)

            if self.board.player == -1:
                self.cum_loss1 += loss
            else:
                self.cum_loss2 += loss

            if i % self.gradient_interval == 0 or i % self.gradient_interval == 1:
                if self.board.player == -1:
                    self.cum_loss1.backward()
                    # for param in policy_net1.parameters():
                    #   param.grad.data.clamp_(-1, 1)
                    self.optimizer1.step()
                    self.cum_loss1 = 0
                    if self.use_memory:
                        self.memory1 = self.memory1.detach()
                else:
                    self.cum_loss2.backward()
                    # for param in policy_net2.parameters():
                    #    param.grad.data.clamp_(-1, 1)
                    self.optimizer2.step()
                    self.cum_loss2 = 0
                    if self.use_memory:
                        self.memory2 = self.memory2.detach()

            # VALIDATION AND LOGGING

            self.mean_loss += loss.item()

            # print(board.n_moves)
            # mean_ratio_board += (board.n_moves.float().mean() / 42.0)
            # mean_error_game += (1.0 * (R==-2)).mean()
            # if i % 10_000 > 9_900:
            #         print(board.state[0])
        # Update lists for training
        self.list_S.append(S)
        self.list_R.append(R)
        self.list_F.append(F)

    def train(self):
        for i in range(0, 1_000_000_000, self.batch_size):
            self.train_move(i)
            if self.do_each_n(i, self.interval_tensorboard):
                self.report(i)
            if self.do_each_n(i, self.validation_interval):
                self.update_target(i)






    def save_models(self, i):
        self.save_model(self.policy_net1, "model", i)
        self.save_model(self.policy_net2, "model-adv", i)


    def report(self, i):
        super().report(i)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i * self.batch_size / self.EPS_DECAY)
        self.writer.add_scalar("eps_threshold", eps_threshold, i)
        self.mean_loss = 0
        if self.use_memory:
            print('mem1', self.memory1)
            print('mem2', self.memory2)


    def update_target(self, i):
        if self.replace_target_if_better:
            for j, policy, target, max_score in zip([1, 2],
                                                    [self.policy_net1, self.policy_net2],
                                                    [self.target_net1, self.target_net2],
                                                    [self.max_score1, self.max_score2]):
                policy.eval()
                player = NNPlayer(policy)
                summary = mirror_score(player, nbatch=self.batch_size, n_iter=200, second_player=trainer_player, device=self.device)
                print(i, j, summary)
                for key, value in summary.items():
                    self.writer.add_scalar(f"{key}_{j}", value, i)

                if j == 1:
                    self.max_score1 = max(summary["score"], max_score)
                    if summary["score"] > max_score:
                        print("Target model updated for ", j)
                        self.target_net1.load_state_dict(policy.state_dict())
                else:
                    self.max_score2 = max(summary["score"], max_score)
                    if summary["score"] > max_score:
                        print("Target model updated for ", j)
                        self.target_net2.load_state_dict(policy.state_dict())
        else:
            self.target_net1.load_state_dict(self.policy_net1.state_dict())
            self.target_net2.load_state_dict(self.policy_net2.state_dict())
        self.save_models(i)


if __name__ == "__main__":
    from model.model_helper import load_model

    hyperparams = {
        "GAMMA": 0.9,
        "EPS_START": 0.9,
        "EPS_END": 0.2,
        "EPS_DECAY": 30_000_000,
        "TARGET_UPDATE": 10, # no more used
        "start_random": 0.9, # randomness modifier at the beginning of each match
        "end_random": 0.5, # randomness modifier at the end of each match
        "decay_random": 10, # randomness modifier decay
        "ratio_trainer": 0.5, #ratio of trainer moves respect to random moves
        "lr": 0.01,
        "interval_tensorboard": 20_000, # every each moves does it pllot to tensorboard
        "validation_interval": 250_000, # every each moves policies are validated
        "replace_target_if_better": True, # Target model is replaced by current policy onli if the score is higher
        "gradient_interval" : 5 # Every each moves gradients are calculated (for memory model otherwise 1)
    }
    device = torch.device("cuda")
    frank_path = "best_trained/ConvNetNoMem/model_793620001.pth"
    frank = load_model(frank_path, ConvNetNoMem, device=device)
    #trainer_player = GreedyModel()
    trainer_player = frank
    model = ConvNet
    trainer = Trainer(batch_size=10000, hyperparams=hyperparams, model=model, trainer_player=trainer_player, seed=99)
    trainer.train()


#type "tensorboard --logdir=runs" in terminal
# 192.1.10.235