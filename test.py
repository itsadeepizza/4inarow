import numpy as np
import torch
import os
import random
from model.model_helper import TreePlayer, play_until_end, AIPlayer, NNPlayer, load_model
from game.board import BatchBoard


def play_hvsm_game(AIplayer, verbose=False):
    """Simulate a game between human player and model"""
    from game.board import BatchBoard
    batchboard = BatchBoard()

    while True:
        print(batchboard)
        player = batchboard.player
        while True:
            print(f"PLAYER {player} - HUMAN")
            if verbose:
                print(AIplayer.get_scores(batchboard))
            print("Choose a valid column:")
            col = torch.tensor([int(input())])
            if batchboard.play(col):
                break
        if batchboard.check_win(player):
            print(batchboard)
            print(f"Player {player} won !")
            return True


        print(batchboard)
        player = batchboard.player

        print(f"PLAYER {player} - MACHINE")

        adv_move = AIplayer.play(batchboard, verbose)

        print(f"Player {player} played {adv_move}")
        if not batchboard.play(adv_move):
            print("NOT VALID MOVE")
            break
        if batchboard.check_win(player):
            print(batchboard)
            print(f"Player {player} won !")
            return False




# Raccogliere i risultati delle partite su una matrice
# Fare una somma delle colonne per stilare una classifica



if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.model import DQN, smallDQN, channel_DQN, full_channel_DQN, ConvNetNoMem, ConvNetNoGroup7
    from model.model_helper import TreePlayer
    from model.greedy_model import GreedyModel
    from validation import mirror_score


    path = f"runs/fit/20220320-001710/models/model_8000.pth"
    path = "runs/fit/20220320-004727/models/model-adv_130000.pth"
    path = "runs/models/model_926000.pth"
    # path = "runs/fit/20220320-004727/models/model_127000.pth"
    path = "runs/fit/20220320-004727/models/model_208000.pth"
    frank_path = "frank/models/model_793620001.pth"
    big_first = "best_trained/ConvNetNoGroup7/model_339370001.pth"
    # AIplayer = TreePlayer(load_model(big_first, ConvNetNoGroup7))
    AIplayer = load_model(big_first, ConvNetNoGroup7)

    #print(mirror_score(AIplayer, nbatch=1000))
    play_hvsm_game(AIplayer, verbose=True)
    from validation import mirror_score

