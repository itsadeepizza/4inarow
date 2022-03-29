import torch
import torch.nn as nn
from copy import deepcopy
#from ..game.board import BatchBoard
class AIPlayer():

    def get_scores(self, batch_board):
        return None

    def play(self, batch_board, verbose=False):
        Q = self.get_scores(batch_board)
        if verbose:
            print(Q)
        move = Q.argmax(dim=1)
        return move


class NNPlayer(AIPlayer):
    def __init__(self, model: nn.Module, use_memory=False):
        self.model = model
        self.use_memory = use_memory

    def get_scores(self, batch_board, memory=None):
        with torch.no_grad():
            if self.use_memory:
                Q, _ = self.model.forward(batch_board.state, memory)
            else:
                Q = self.model.forward(batch_board.state)
        return Q


class TreePlayer(AIPlayer):
    """Convert a base player to a tree player"""

    def __init__(self, base_player: AIPlayer):
        self.base_player = base_player


    def get_scores(self, batch_board, memory=None):
        """Test all alternatives for each column, simulating the game"""
        # Step 1: Test all columns
        # Step 2: For each column, calculate the reward
        # Step 3: Keep the move with the highest reward
        # Make a board for each possibility
        device = batch_board.device
        all_test = [deepcopy(batch_board) for i in range(batch_board.ncols)]
        # play a different column for each copy of the board
        score = self.base_player.get_scores(batch_board)
        for i, case_batch_board in enumerate(all_test):
            played_cols = torch.ones([case_batch_board.nbatch], device=device) * i
            mult = case_batch_board.player
            is_valid = case_batch_board.play(played_cols)
            score[~ is_valid, i] -= 100
            # TODO: add number of moves before loose
            results = play_until_end(self.base_player, self.base_player, case_batch_board, verbose=False)
            score[:, i] += 10 * results * mult

        return score

def play_until_end(player1: AIPlayer, player2: AIPlayer, batch_board, verbose=False):
    device = batch_board.device
    all_finished = torch.zeros([batch_board.nbatch], device=device).bool()
    results = torch.zeros([batch_board.nbatch], device=device)
    while not torch.all(all_finished):
        if verbose:
            print(batch_board)
        if batch_board.player == 1:
            player = player1
        else:
            player = player2
        moves = player.play(batch_board, verbose=verbose)
        summary = batch_board.get_reward(moves)
        results[summary["has_win"] & ~ all_finished] = - batch_board.player
        results[~summary["is_valid"] & ~ all_finished] = batch_board.player
        all_finished[summary["is_final"]] = True
    return results


def load_model(path, model_class, device=torch.device("cpu")):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return NNPlayer(model)