import numpy as np
class Board:
    def __init__(self, nrows=6, ncols=7):
        # 0 no coin, otherwise plaYEr number coin
        self.nrows = nrows
        self.ncols = ncols
        self.board = np.zeros([nrows, ncols])
        # row level for each col
        self.cols = np.zeros([ncols], dtype=int)
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
        #change player
        self.player *= -1
        # mark the new row in cols
        self.cols[col] += 1
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








board = Board()
board.play(1)
board.play(2)
board.play(2)
board.play(3)
board.play(3)
board.play(4)
board.play(3)
board.play(4)
board.play(4)
board.play(6)
board.play(4)
print(board)
print(board.check_win(-1))
board = Board()
board.play_hvsh_game()