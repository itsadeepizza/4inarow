from model.model import AIPlayer
from game.board import BatchBoard
import torch


def mirror_score(player: AIPlayer, nbatch=1, n_iter=200, rand_ratio=0.2, cols=7, device=torch.device("cpu")):
    """Make the model play against a randomized version of itself"""
    batch_board = BatchBoard(nbatch=nbatch, device=device)
    n_match = win = lost = error = 0
    for i in range(n_iter):
        # Not randomized player
        move = player.play(batch_board)
        is_final, rewards, adv_rewards, has_win, is_valid = batch_board.get_reward(move)
        n_match += is_final.sum().item()
        win += has_win.sum().item()
        error += (~is_valid).sum().item()

        # Randomized player

        move = player.play(batch_board)
        # Choose a random move with rand_ratio probability
        rand_move = torch.randint(0, cols, [nbatch], device=device)
        # When choosing a random move (or a greedy move)
        rand_choice = torch.rand([nbatch], device=device)
        move[rand_choice < rand_ratio] = rand_move[rand_choice < rand_ratio]
        # play the move
        is_final, rewards, adv_rewards, has_win, is_valid = batch_board.get_reward(move)
        n_match += is_final.sum().item()
        lost += has_win.sum().item()
    summary = {
        "n_match": n_match,
        "average_len": (nbatch * n_iter) / n_match,
        "ratio_error": error / n_match,
        "ratio_win": win / n_match,
        "ratio_lost": lost / n_match,
        "score": win / (lost + error)
    }




    return summary




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.model import DQN, smallDQN, channel_DQN, full_channel_DQN, ConvNet, NNPlayer
    from model.greedy_model import GreedyModel
    import glob
    model = full_channel_DQN()
    path = f"runs/fit/20220227-104623/models/model-adv_2040000.pth"
    model = ConvNet()
    path = f"runs/models/model_50000.pth"
    dir = "runs/models"
    for path in glob.glob(dir + "/*.pth"):
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(device)
        player = NNPlayer(model)

    #player = GreedyModel()
        summary = mirror_score(player, nbatch=1000, device=device, rand_ratio=0.2)
        print(path)
        print(summary)
        print("-" * 20)