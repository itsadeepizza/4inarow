import numpy as np
import torch
import os
import random
from model.model_helper import TreePlayer, play_until_end, AIPlayer, NNPlayer, load_model
from game.board import BatchBoard
from functools import lru_cache


def models_in_directory(directory, model_class):
    result = []
    for root, dirs, files in os.walk(directory):
        for filepath in files:
            if filepath.endswith(".pth"):
                result.append({"filepath": root+"/"+filepath, "model": model_class})
        for d in dirs:
            for filepath in files:
                if filepath.endswith(".pth"):
                    result.append({"filepath": root+"/"+d+"/"+filepath, "model": model_class})
    return result


def models_in_all_directories(input_list):
    # example input: [{"dir": "runs/fit/23984298/models", "model": ConvNet}, {"dir": "runs/fit/98624986/models", "model": SmallDQN}]
    # example output: [{"weights": "runs/fit/23984298/models/8937289273.pth", "model": ConvNet}, {"weights": "runs/fit/98624986/models/297390273.pth", "model": SmallDQN}, ...]

    model_list = []
    for element in input_list:
        model_list += models_in_directory(element["dir"], element["model"])
    return model_list


def get_players(models_list, i1, i2, device=torch.device("cpu")):
    model1 = models_list[i1]
    model2 = models_list[i2]
    player1 = load_model(model1["filepath"], model1["model"], device=device)
    player2 = load_model(model2["filepath"], model2["model"], device=device)
    return player1, player2


@lru_cache
def all_combinations(ncols=7, nmoves=3, device=torch.device("cpu")):
    import itertools
    t = torch.tensor(list(itertools.product(range(ncols), repeat=nmoves)))
    t = t.to(device)
    return t


def AIvsAI(player1: AIPlayer, player2: AIPlayer, ncols=7, device=torch.device("cpu")):
    """Choose all possible combinations for the first three moves and make two players
    play one against the other """
    batch_size = ncols ** 3
    batch_board = BatchBoard(nbatch=batch_size, device=device)
    all_first_three_moves =  all_combinations(7, 3, device=device)
    initial_moves = all_first_three_moves[:, 0]
    batch_board.play(initial_moves)
    second_moves = all_first_three_moves[:, 1]
    batch_board.play(second_moves)
    third_moves = all_first_three_moves[:, 2]
    batch_board.play(third_moves)
    results = play_until_end(player1, player2, batch_board, verbose=False)
    return results.sum() / batch_size


def hunger_games(models_list, n_match=10, device=torch.device("cpu")):
    from tqdm import tqdm
    from itertools import permutations
    import random
    n_models = len(models_list)
    score_matrix = torch.zeros([n_models, n_models])
    time_played = torch.zeros([n_models])
    if n_models < 2:
        raise Exception("not enough models")
    #n_match = min(n_match, n_models * (n_models - 1))
    all_couples = list(permutations(range(n_models), 2))
    random.shuffle(all_couples)
    selected_couples = all_couples[:n_match]
    for i1, i2 in tqdm(selected_couples):
        player1, player2 = get_players(models_list, i1, i2, device=device)
        score_matrix[i1, i2] = AIvsAI(player1, player2, device=device)
        time_played[i1] += 1
    return score_matrix, time_played


def leader_board(input_list, n_match=10, device=torch.device("cpu")):
    """Make a leader bord starting from a list of directories and associated models in the form:
    [{"dir": "runs/fit/23984298/models", "model": ConvNet}, {"dir": "runs/fit/98624986/models", "model": SmallDQN}]
   the output is a dictionary with all scores related to each model"""
    models_list = models_in_all_directories(input_list)
    score_matrix, time_played = hunger_games(models_list, n_match, device=device)
    scores = score_matrix.sum(axis=1) / time_played
    leader_board = {model["filepath"]: score.item() for model, score in zip(models_list, scores) if not torch.isnan(score)}
    return leader_board



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model.model import ConvNet, ConvNetNoMem
    input = [{"dir": "./saves/ConvNet", "model": ConvNetNoMem}, {"dir": "./saves/ConvNetMem", "model": ConvNet}]
    input = [{"dir": "runs/fit/20220320-004727/models", "model": ConvNetNoMem}]
    lb = leader_board(input, 1000000, device=device)
    import pickle
    print(lb)
    with open("leaderboard3.txt", "w") as f:
        f.write(str(lb))