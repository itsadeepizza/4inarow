import numpy as np
import torch
import os
import random
from model.model_helper import TreePlayer, play_until_end, AIPlayer, NNPlayer
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

# Saverio
# Creare una funzione che estrae tutti i modelli presenti nelle varie directory, in coppia col modello associato (se di cardibnalità inferiore a 2 restituisce errore)
# selezionare tutti i modelli nelle varie directory


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

def get_players(models_list):
    index1 = random.randint(0,len(models_list) - 1)
    index2 = index1
    while index2 == index1:
        index2 = random.randint(0, len(models_list) - 1)
    model1 = models_list[index1]
    model2 = models_list[index2]
    player1 = load_model(model1["filepath"], model1["model"])
    player2 = load_model(model2["filepath"], model2["model"])
    models = (index1, player1, index2, player2)
    return models

# Paolo
# Far giocare delle coppie di modelli scelti a caso tra loro

def hunger_games(models_list, n_match=10, device=torch.device("cpu")):
    from tqdm import tqdm
    n_models = len(models_list)
    score_matrix = torch.zeros([n_models, n_models])
    time_played = torch.zeros([n_models])
    if n_models < 2:
        raise Exception("not enough models")
    n_match = min(n_match, 2 * n_models * (n_models - 1))
    already_played = {(0,0)}
    i1 = i2 = 0
    for i in tqdm(range(n_match)):
        while (i1, i2) in already_played:
            i1, player1, i2, player2 = get_players(models_list)
        already_played.add((i1, i2))
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
    leader_board = {model["filepath"]: score.item() for model, score in zip(models_list, scores) if score.item() > 0}
    return leader_board


# Creare una funzione che faccia giocare due modelli m'uno contro l'altro
# AIvsAI



def all_combinations(ncols=7, nmoves=3):
    import itertools
    return torch.tensor(list(itertools.product(range(ncols), repeat=nmoves)))

all_first_three_moves = all_combinations(7, 3)

def AIvsAI(player1: AIPlayer, player2: AIPlayer, ncols=7, device=torch.device("cpu")):
    """Choose all possible combinations for the first three moves and make two players
    play one against the other """
    batch_size = ncols ** 3
    batch_board = BatchBoard(nbatch=batch_size)
    initial_moves = all_first_three_moves[:, 0]
    batch_board.play(initial_moves)
    second_moves = all_first_three_moves[:, 1]
    batch_board.play(second_moves)
    third_moves = all_first_three_moves[:, 2]
    batch_board.play(third_moves)
    results = play_until_end(player1, player2, batch_board, verbose=False)
    return results.sum() / batch_size


# Raccogliere i risultati delle partite su una matrice
# Fare una somma delle colonne per stilare una classifica

def load_model(path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    return NNPlayer(model)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.model import ConvNet, ConvNetNoMem
    input = [{"dir": "./saves/ConvNet", "model": ConvNetNoMem}, {"dir": "./saves/ConvNetMem", "model": ConvNet}]
    input = [{"dir": "runs/fit/20220320-004727/models", "model": ConvNetNoMem}]
    print(leader_board(input, 20, device=device))