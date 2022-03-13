import numpy as np
import torch


def play_hvsm_game(AIplayer):
    """Simulate a game between human player and model"""
    from game.board import BatchBoard
    batchboard = BatchBoard()

    while True:
        print(batchboard)
        player = batchboard.player
        while True:
            print(f"PLAYER {player} - HUMAN")
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

        adv_move = AIplayer.play(batchboard)

        print(f"Player {player} played {adv_move}")
        if not batchboard.play(adv_move):
            print("NOT VALID MOVE")
            break
        if batchboard.check_win(player):
            print(batchboard)
            print(f"Player {player} won !")
            return False





if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.model import DQN, smallDQN, channel_DQN, full_channel_DQN, ConvNet, NNPlayer
    from model.greedy_model import GreedyModel





    model = ConvNet()
    path = f"../runs/models/model_200000.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    AIplayer = NNPlayer(model)

    AIplayer = GreedyModel()
    # policy_net = torch.load(path)


    play_hvsm_game(AIplayer)
