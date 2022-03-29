import torch

from model.model import ConvNetNoMem, ConvNetNoGroup7
from model.model_helper import NNPlayer, load_model
from model.greedy_model import GreedyModel
from leaderboard import AIvsAI

if __name__ == "__main__":
    device = torch.device("cuda")
    big_first = "best_trained/ConvNetNoGroup7/model_339370001.pth"
    big_2 = "best_trained/ConvNetNoGroup7/model_605120001.pth"
    # AIplayer = TreePlayer(load_model(big_first, ConvNetNoGroup7))
    pl1 = load_model(big_first, ConvNetNoGroup7, device=device)
    pl3 = load_model(big_2, ConvNetNoGroup7, device=device)
    greedy = GreedyModel()
    little_last = "best_trained/ConvNetNoMem/model_793620001.pth"
    pl2 = load_model(little_last, ConvNetNoMem, device=device)

    print(AIvsAI(pl1, pl3, device=device))