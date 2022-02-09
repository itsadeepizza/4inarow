import numpy as np
import torch
class Board:
    def __init__(self, nrows=6, ncols=7):
        # 0 no coin, otherwise plaYEr number coin
        self.nrows = nrows
        self.ncols = ncols
        self.reinitialize()
    @property
    def state(self):
        """return board, but coin marked 1 are from current player"""
        return torch.FloatTensor(self.board * self.player)

    def __repr__(self):
        def get_symbol(n):
            if n==0:
                return " "
            if n==1:
                return "X"
            if n==-1:
                return "O"
        out = ""
        for row in self.board:
            out_row = ""
            for cell in row:
                symb = get_symbol(cell)
                out_row += f"|{symb}"
            out = out_row + "|\n" + out
        for i in range(self.ncols):
            out += "*" + str(i)
        out += "*"
        return out

    def play(self, col):
        """Put a coin in the `col` column. Return False if impossible"""
        if (col >= self.ncols) or (col < 0):
            # Error if no column
            return False
        curr_row = self.cols[col]
        if curr_row>= self.nrows:
            # error if column is full
            return False
        # put the coin for current player
        self.board[curr_row][col] = self.player
        # mark the new row in cols
        self.cols[col] += 1
        # change player
        self.player *= -1
        # If no error
        return True

    def check_win(self, player):
        """return True if the player `player` has won"""

        def check_win_line(board, player):
            """Check if player has won in a linearized array"""
            for i in range(len(board) - 3):
                fours = board[i: i + 4]
                if np.all(fours == player):
                    return True
            return False

        # add empty rows and columns
        check_board = np.vstack((self.board, np.zeros([1, self.ncols])))
        check_board = np.hstack((check_board, np.zeros([self.nrows + 1, 1])))
        # flatten for each possible direction
        is_hor_win = check_win_line(check_board.flatten(), player)
        is_ver_win = check_win_line(check_board.transpose().flatten(), player)
        is_diag1_win = check_win_line(np.array([np.roll(row, i) for i, row in enumerate(check_board)]).transpose().flatten(), player)
        is_diag2_win = check_win_line(np.array([np.roll(row, - i) for i, row in enumerate(check_board)]).transpose().flatten(), player)
        # has player won?
        win = is_hor_win or is_ver_win or is_diag1_win or is_diag2_win
        # print(is_hor_win, is_ver_win, is_diag1_win, is_diag2_win)
        return win

    def play_hvsh_game(self):
        """Simulate a game between two human players"""
        while True:
            print(self)
            player = self.player
            while True:
                print(f"PLAYER {player}")
                print("Choose a valid column:")
                col = int(input())
                if self.play(col):
                    break
            if self.check_win(player):
                print(f"Player {player} won !")
                return True


            print(self)
            player = self.player
            while True:
                print(f"PLAYER {player}")
                print("Choose a valid column:")
                col = int(input())
                if self.play(col):
                    break
            if self.check_win(player):
                print(f"Player {player} won !")
                return False

    def play_hvsm_game(self):
        """Simulate a game between human player and model"""
        from model.model import DQN

        policy_net = DQN()
        path = "../runs/model_180000.pt"
        policy_net.load_state_dict(torch.load(path))
        #policy_net = torch.load(path)
        policy_net.eval()

        while True:
            print(self)
            player = self.player
            while True:
                print(f"PLAYER {player} - HUMAN")
                print("Choose a valid column:")
                col = int(input())
                if self.play(col):
                    break
            if self.check_win(player):
                print(self)
                print(f"Player {player} won !")
                return True


            print(self)
            player = self.player

            print(f"PLAYER {player} - MACHINE")

            adv_state = self.state.unsqueeze(0)

            with torch.no_grad():
                Q = policy_net.forward(adv_state)
                print(Q)
                adv_move = torch.argmax(Q).item()
            if not self.play(adv_move):
                print("NOT VALID MOVE")
                break
            if self.check_win(player):
                print(self)
                print(f"Player {player} won !")
                return False

    def reinitialize(self):
        """Reinitialise the grid"""
        self.board = np.zeros([self.nrows, self.ncols])
        # row level for each col
        self.cols = np.zeros([self.ncols], dtype=int)
        # player are 1 and -1
        self.player = 1

    def get_reward(self, col):
        """Return the reward associated to the move
        1 if win
        -1 if lose
        -2 invalid play
        0 otherwise

        and a boolean which is true if a new game is starting
        """
        rew_win = 1
        rew_lose = -1
        rew_invalid = -2
        if self.check_win(- self.player):
            self.reinitialize()
            return rew_lose, True
        is_valid = self.play(col)
        if not is_valid:
            self.reinitialize()
            return rew_invalid, True
        if self.check_win(- self.player):
            return rew_win, False
        return 0, False

class BatchBoard:
    def __init__(self, nbatch=1, nrows=6, ncols=7, device=torch.device("cpu")):
        # 0 no coin, otherwise plaYEr number coin
        self.nrows = nrows
        self.ncols = ncols
        self.nbatch = nbatch
        self.device = device
        self.reinitialize()

    def reinitialize(self):
        """Reinitialise the grid"""
        self.board = torch.zeros([self.nbatch, self.nrows, self.ncols], device=self.device)
        # row level for each col
        self.cols = torch.zeros([self.nbatch, self.ncols], dtype=int, device=self.device)
        # player are 1 and -1
        self.player = 1

    @property
    def state(self):
        """return board, but coin marked 1 are from current player"""
        return self.board * self.player

    def __repr__(self):
        def get_symbol(n):
            if n==0:
                return " "
            if n==1:
                return "X"
            if n==-1:
                return "O"
        out = ""
        for batch in self.board:
            out_batch = ""
            for row in batch:
                out_row = ""
                for cell in row:
                    symb = get_symbol(cell)
                    out_row += f"|{symb}"
                out_batch = out_row + "|\n" + out_batch
            for i in range(self.ncols):
                out_batch += "*" + str(i)
            out_batch += "*\n"
            out += out_batch
        return out

    def play(self, col: torch.Tensor):
        """Put a coin in the `col` column. Return False if impossible"""
        # Create a sparse matrix for selecting chosen columns for each batch
        indx = torch.stack((torch.arange(self.nbatch, device=self.device),
                           col))
        vals = torch.ones([self.nbatch], device=self.device)
        mask = torch.sparse_coo_tensor(indx,
                                       vals,
                                       (self.nbatch, self.ncols),
                                       dtype=bool,
                                       device = self.device).to_dense()
        # Check if the chosen columns are out of range
        # is_valid = (col > 0) & (col < self.ncols)
        # Check if the chosen columns are already filled
        curr_row = self.cols[mask]
        is_valid = curr_row < self.nrows # & is_valid
        # put the coin for current player
        #self.board[curr_row][col] = self.player
        # add 1 to each chosen columns for each batch
        board_indx = torch.stack((torch.arange(self.nbatch, device=self.device),
                            curr_row,
                            col))
        # keep only valid columns
        board_indx = board_indx[:, is_valid]
        board_vals = torch.ones([is_valid.sum()], device=self.device)
        board_mask = torch.sparse_coo_tensor(board_indx,
                                       board_vals,
                                       (self.nbatch, self.nrows, self.ncols),
                                       dtype=bool,
                                       device=self.device).to_dense()
        self.board[board_mask] = self.player
        self.cols[mask] += 1
        # change player
        self.player *= -1
        # return batch with valid move
        return is_valid


    def check_draw(self):
        """return True if the board is full and no move can be played"""
        return self.cols.min(1).values >= self.nrows

    def check_win(self, player):
        """return True if the player `player` has won"""

        def check_win_line(line, player):
            """Check if player has won in a batch of linearized arrays"""
            four_adjacent = (line[:, :-3] == player) & \
                            (line[:, 1:-2] == player) & \
                            (line[:, 2:-1] == player) & \
                            (line[:, 3:] == player)
            player_win = four_adjacent.any(1)
            return player_win

        def roll_by_gather(mat, sgn):
            """shift all rows of different batches of incremental value toward left or right according to sgn = 1 or -1"""
            # assumes 2D array
            n_batch, n_rows, n_cols = mat.shape

            a1 = torch.arange(n_rows, device=mat.device).view((1, n_rows, 1)).repeat((n_batch, 1, n_cols))
            a2 = (sgn * (torch.arange(n_cols, device=mat.device) - a1)) % n_cols
            return torch.gather(mat, 2, a2 )


        # add empty rows and columns
        z1 = torch.zeros([self.nbatch, 1, self.ncols], device=self.device)
        check_board = torch.cat((self.board, z1), dim=1)
        z2 = torch.zeros([self.nbatch, self.nrows + 1, 1], device=self.device)
        check_board = torch.cat((check_board, z2), dim=2)

        # flatten for each possible direction
        is_hor_win = check_win_line(check_board.flatten(1), player)
        is_ver_win = check_win_line(check_board.permute((0,2,1)).flatten(1), player)
        is_diag1_win = check_win_line(roll_by_gather(check_board, 1).permute((0,2,1)).flatten(1), player)
        is_diag2_win = check_win_line(roll_by_gather(check_board, -1).permute((0,2,1)).flatten(1), player)
        # has player won?
        win = is_hor_win | is_ver_win | is_diag1_win | is_diag2_win
        #print(is_hor_win, is_ver_win, is_diag1_win, is_diag2_win)
        return win


    def get_reward(self, cols):
        """Return the reward associated to the move
        1 if win
        -1 if lose
        -2 invalid play
        0 otherwise

        for the player and the opponent, and if it is the final state of the game
        """
        rew_win = 1
        rew_lose = -1
        rew_invalid = -2
        rew_draw = 0

        rewards = torch.zeros([self.nbatch], device=self.device)
        adv_rewards = torch.zeros([self.nbatch], device=self.device)
        is_valid = self.play(cols)
        # negative rewards for invalid moves
        rewards[~ is_valid] = rew_invalid
        has_win = self.check_win(- self.player)
        # positive rewards for winner
        rewards[has_win] = rew_win
        # negative rewards for loser (the opponent)
        adv_rewards[has_win] = rew_lose
        # zero rewards for draw game (player and opponent)
        is_draw = self.check_draw()
        rewards[is_draw] = rew_draw
        adv_rewards[is_draw] = rew_draw
        is_final = has_win | (~is_valid) | is_draw
        #reinitialise game if it is a final state
        self.board[is_final] = 0
        self.cols[is_final] = 0
        return is_final, rewards, adv_rewards


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board = BatchBoard(nbatch=2, device=device)
    cols = torch.tensor([3, 3]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([3, 4]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([2, 4]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([2, 5]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([1, 5]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([2, 6]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([0, 5]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([2, 6]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([1, 6]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([2, 4]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))

    cols = torch.tensor([1, 6]).to(device)
    print(board.get_reward(cols))
    print(board)
    print(board.check_win(1))
    print(board.check_win(-1))

    board = Board()
    board.play_hvsm_game()
