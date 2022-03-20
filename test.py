import numpy as np
import torch


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





if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.model import DQN, smallDQN, channel_DQN, full_channel_DQN, ConvNet
    from model.greedy_model import GreedyModel
    from model.model_helper import TreePlayer, play_until_end, NNPlayer

    def load_model(model_class, path):
        model = model_class()
        model.load_state_dict(torch.load(path))
        model.eval()
        return NNPlayer(model)


    path = f"runs/fit/20220320-001710/models/model_8000.pth"
    path_1 = "runs/fit/20220320-004727/models/model-adv_130000.pth"
    path_bruno = "runs/models/model_926000.pth"
    #path = "runs/fit/20220320-004727/models/model_127000.pth"
    path_2 = "runs/fit/20220320-004727/models/model-adv_321000.pth"
    path_12 = "runs/fit/20220320-004727/models/model_286000.pth"

    AIplayer = load_model(ConvNet, path_bruno)

    AIplayer2 = GreedyModel()
    AIplayer3 = load_model(ConvNet, path_1)
    AIplayer4 = load_model(ConvNet, path_2)
    # policy_net = torch.load(path)


    from game.board import BatchBoard
    tP = TreePlayer(AIplayer3)
    #tP = TreePlayer(tP)
    res = play_until_end(AIplayer, tP,  BatchBoard(1), True)
    print(res)
    play_hvsm_game(tP, verbose=True)
