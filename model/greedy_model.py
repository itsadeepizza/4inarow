from game.board import BatchBoard
from model.model import AIPlayer
import torch
from copy import deepcopy

class GreedyModel(AIPlayer):

    def get_scores(self, batch_board: BatchBoard):
        """An algorithm which maximise short term reward"""
        # Step 1: Test all columns
        # Step 2: For each column, calculate the reward
        # Step 3: Keep the move with the highest reward
        # Make a board for each possibility
        device = batch_board.device
        all_test = [deepcopy(batch_board) for i in range(batch_board.ncols)]
        # play a different column for each copy of the board
        all_rewards = []
        for i, batch_board in enumerate(all_test):
            played_cols = torch.ones([batch_board.nbatch], device=device) * i
            _, rewards, _, _, _ = batch_board.get_reward_v2(played_cols)
            all_rewards.append(rewards)
        # For each board in the batch, keep the column associated to the best reward
        all_rewards = torch.stack(all_rewards)
        random_modifier = torch.rand_like(all_rewards, device=device) * 0.0001
        all_rewards += random_modifier

        all_rewards = torch.swapaxes(all_rewards, 0, 1)
        return all_rewards





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = BatchBoard(nbatch=2, device=device)
    cols = torch.tensor([3, 3]).to(device)
    #print(board.get_reward(cols))

    print(greedy_player(board))

