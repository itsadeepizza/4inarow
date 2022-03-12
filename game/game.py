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

        adv_state = batchboard.state

        with torch.no_grad():
            Q = policy_net.forward(adv_state)
            print(Q)
            adv_move = Q.argmax(dim=1)
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

    model = full_channel_DQN()
    path = f"../runs/fit/20220227-104623/models/model-adv_2040000.pth"
    model.load_state_dict(torch.load(path))
    # policy_net = torch.load(path)
    model.eval()

    play_hvsm_game(model)
