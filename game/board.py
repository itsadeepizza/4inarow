import torch


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
        # number og moves already played
        self.n_moves = torch.zeros([self.nbatch], dtype=int, device=self.device)



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
        # increment counter number of moves
        self.n_moves += 1
        # Create a sparse matrix for selecting chosen columns for each batch
        indx = torch.stack((torch.arange(self.nbatch, device=self.device),
                           col))
        vals = torch.ones([self.nbatch], device=self.device)
        mask = torch.sparse_coo_tensor(indx, # RuntimeError: "coalesce" not implemented for 'Bool' <- samas triste :(
                                       vals,
                                       (self.nbatch, self.ncols),
                                       device = self.device).to_dense().bool() # Error fixed adding .bool() at the end
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
                                       device=self.device).to_dense().bool()
        self.board[board_mask] = self.player
        self.cols[mask] += 1
        # change player
        self.player *= -1
        # return batch with valid move
        return is_valid


    def check_draw(self):
        """return True if the board is full and no move can be played"""
        return self.cols.min(1).values >= self.nrows

    def count_triplet(self, player):
        """return True if the player `player` has won"""

        def count_triplet_line(line, player):
            """Check if player has won in a batch of linearized arrays"""
            three_adjacent_begin = (line[:, :-3] == player) & \
                            (line[:, 1:-2] == player) & \
                            (line[:, 2:-1] == player) & \
                            (line[:, 3:] == 0)
            three_adjacent_end = (line[:, :-3] == 0) & \
                                   (line[:, 1:-2] == player) & \
                                   (line[:, 2:-1] == player) & \
                                   (line[:, 3:] == player)
            number_triplet = three_adjacent_begin.sum(axis=1) + three_adjacent_end.sum(axis=1)
            return number_triplet

        def roll_by_gather(mat, sgn):
            """shift all rows of different batches of incremental value toward left or right according to sgn = 1 or -1"""
            # assumes 2D array
            n_batch, n_rows, n_cols = mat.shape

            a1 = torch.arange(n_rows, device=mat.device).view((1, n_rows, 1)).repeat((n_batch, 1, n_cols))
            a2 = (sgn * (torch.arange(n_cols, device=mat.device) - a1)) % n_cols
            return torch.gather(mat, 2, a2)

        # add empty rows and columns
        z1 = torch.ones([self.nbatch, 1, self.ncols], device=self.device) * 2
        check_board = torch.cat((self.board, z1), dim=1)
        z2 = torch.ones([self.nbatch, self.nrows + 1, 1], device=self.device) * 2
        check_board = torch.cat((check_board, z2), dim=2)

        # flatten for each possible direction
        n_hor_triplet = count_triplet_line(check_board.flatten(1), player)
        n_ver_triplet = count_triplet_line(check_board.permute((0, 2, 1)).flatten(1), player)
        n_diag1_triplet = count_triplet_line(roll_by_gather(check_board, 1).permute((0, 2, 1)).flatten(1), player)
        n_diag2_triplet = count_triplet_line(roll_by_gather(check_board, -1).permute((0, 2, 1)).flatten(1), player)
        # has player won?
        tot_triplet = n_hor_triplet + n_ver_triplet + n_diag1_triplet + n_diag2_triplet
        # print(is_hor_win, is_ver_win, is_diag1_win, is_diag2_win)
        return tot_triplet

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
        rew_none = 0

        rewards = torch.ones([self.nbatch], device=self.device) * rew_none
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
        self.n_moves[is_final] = 0
        summary = {
            "is_final": is_final,
            "rewards": rewards,
            "adv_rewards": adv_rewards,
            "has_win": has_win,
            "is_valid": is_valid,
            "is_draw": is_draw
        }
        return summary

    def get_reward_v2(self, cols):
        """Return the reward associated to the move
        10 if win
        1 if seq 2
        2 if seq 3

        # neg reward if enemy closer to win

        -1 if lose
        -2 invalid play
        0 otherwise

        for the player and the opponent, and if it is the final state of the game
        """
        rew_win = 1
        rew_lose = -1
        rew_invalid = -2
        rew_draw = 0
        rew_none = 0

        rewards = torch.ones([self.nbatch], device=self.device) * rew_none
        adv_rewards = torch.zeros([self.nbatch], device=self.device)
        is_valid = self.play(cols)

        # Check if we have blocked an enemy win
        #enemy_triplet = self.count_triplet(self.player)
        player_triplet = self.count_triplet(- self.player)
        adv_player_triplet = self.count_triplet(self.player)

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
        summary = {
            "is_final": is_final,
            "rewards": rewards + player_triplet * 0.2 - adv_player_triplet * 0.3,
            "adv_rewards": adv_rewards - player_triplet * 0.2,
            "has_win": has_win,
            "is_valid": is_valid
        }
        return summary
