import numpy as np
import torch


def play_hvsm_game(model):
    """Simulate a game between human player and model"""
    from board import BatchBoard
    batchboard = BatchBoard()


    policy_net = model


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

        adv_move = model.play(batchboard)

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
    from model.model import DQN, smallDQN, channel_DQN, full_channel_DQN, ConvNet
    from model.greedy_model import GreedyModel


    def make_player(model):
        def play(batchboard):
            with torch.no_grad():
                Q = model.forward(batchboard.state)
                print(Q)
                move = Q.argmax(dim=1)
            return move
        return play


    model = full_channel_DQN()
    model.play = make_player(model)
    path = f"../runs/fit/20220227-104623/models/model-adv_2040000.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    model = GreedyModel()
    # policy_net = torch.load(path)


    play_hvsm_game(model)
