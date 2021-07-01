from time import time

class Minimax:


    def __init__(self, depth, board = None):
        self.MINIMAX_DEPTH = depth

        if board != None:
            self.board = board
        else:
            self.board = [[  1,  1,  1,  1,  1],
                          [  1,  0,  0,  0,  1],
                          [  1,  0,  0,  0, -1],
                          [ -1,  0,  0,  0, -1],
                          [ -1, -1, -1, -1, -1]]

        # self.weights =   [[  1,   1,   1,   1 ,  1],
        #                   [  1, 1.25, 1.2, 1.25,  1],
        #                   [1.1, 1.2,  1.4, 1.2, 1.1],
        #                   [  1, 1.25, 1.2, 1.25,  1],
        #                   [  1,   1,   1,   1,   1]]


    def show_board(self, board = None):
        """ Display a game board

        Parameters:
        -----------
        board : 2D list
            The board you want to display
        """

        if board == None:
            board = self.board
        for i in range(5):
            if i == 0:
                print('x\\y|', end='')
            print("{0:>3}".format(str(i)), end='')
            if i != 4:
                print(',',end='')
        print()

        for i in range(5):
            print("{0:->5}".format("-"), end='')
        print()

        for j in range(5):
            for i in range(5):
                if i == 0:
                    print(f"{j}  |",end='')
                print("{0:>3}".format(str(board[j][i])), end='')
                if i != 4:
                    print(',',end='')
            print()
        print()


    def get_player_pieces(self, board):
        """ 
        Parameters:
        ----------
        board : 2D list
            The board state
        
        Returns:
        --------
        dict
            a dictionary dict where dict[1] is a list of tuple representing the position of cell 1 
            and dict[-1] is a list of tuple representing the position of cell -1
        """

        player_pieces = {1:[], -1:[]}
        for i in range(5):
            for j in range(5):
                if board[i][j] == 1:
                    player_pieces[1].append((i, j))
                elif board[i][j] == -1:
                    player_pieces[-1].append((i, j))
        return player_pieces


    def get_all_nearby_cell(self, x, y):
        """
        Parameters:
        -----------
        x : int
        y : int
            (x, y) is the coordinate of a cell

        Returns:
        --------
        list
            a list that contains position tuples, each represents the position of the cell near the cell at (x, y)
        """

        # cells = []
        # if x < 4:
        #     cells.append((x+1, y))
        # if y < 4:
        #     cells.append((x, y+1))
        # if x > 0:
        #     cells.append((x-1, y))
        # if y > 0:
        #     cells.append((x, y-1))
        # if x + y == 4 or x + y == 2 or x + y == 6:
        #     if x < 4 and y > 0:
        #         cells.append((x+1, y-1))
        #     if x > 0 and y < 4:
        #         cells.append((x-1, y+1))
        # if x - y == 0 or x - y == 2 or y - x == 2:
        #     if x < 4 and y < 4:
        #         cells.append((x+1, y+1))
        #     if x > 0 and y > 0:
        #         cells.append((x-1, y-1))

        ### all cells in square x-1 -> x+1, y-1 -> y+1 except center
        all_cells = [(t1,t2) for t1 in range(x-1,x+2) for t2 in range(y-1,y+2) \
                                if (t1,t2) != (x,y) and t1*(4-t1)>=0 and t2*(4-t2)>=0]
        if (x+y) % 2 == 0:
            cells = all_cells
        else:
            cells = [(i,j) for (i,j) in all_cells if i==x or j==y]
        return cells


    def get_nearby_empty_cell(self, board, x, y):
        """
        Parameters:
        -----------
        board : 2D list
            The board state
        x : int
        y : int
            x, y is the coordinate of the cell

        Returns
        -------
        list
            a list that contains position tuples of empty cells near the cell at (x, y) 
        """

        emp_cells = []
        nearby = self.get_all_nearby_cell(x, y)
        for cell in nearby:
            if board[cell[0]][cell[1]] == 0:
                emp_cells.append(cell)
        return emp_cells
    

    def get_nearby_cells_of_player(self, board, player, x, y):
        """
        Parameters:
        -----------
        board : 2D array
            The board state
        x : int
        y : int
            x, y is the coordinate of the cell
        player : 1 or -1
            the type of player

        Returns:
        --------
        list
            a list that contains position tuples where: board[position] == player and position is near (x, y) 
        """

        result = []
        nearby = self.get_all_nearby_cell(x, y)
        for cell in nearby:
            if board[cell[0]][cell[1]] == player:
                result.append((cell[0], cell[1]))
        return result


    def evaluate_board(self, board, player):
        """
        Parameters:
        -----------
        board : 2D list
            the board state
        player : 1 or -1
            the type of player
        
        Returns:
        --------
        int
            a number represents how favourable a board state is, according to a specific player 
        """

        value = 0.0
        for (i,j) in [(i,j) for i in range(5) for j in range(5)]:
            value += (board[i][j] / player) ### + if same sign, - if diff sign, 0 if not player
                # if board[i][j] == player:
                #     value += self.weights[i][j]
                # elif board[i][j] == -player:
                #     value -= self.weights[i][j]
        return value


    def clone_board(self, board):
        """
        Parameters:
        -----------
        board : 2D list
            the board state

        Returns:
        -------
        2D list
            a deep copy of a board
        """

        new_board = [[elem for elem in row] for row in board]
        return new_board


    def move(self, board, player, trap_move = None):

        """ This function is required in the assignment.

        Parameters:
        -----------
        board : 2D list
            the board state
        player : -1 or 1
            the type of player
        trap_move : tuple(int, int)
            if the opponent's move is a trap move, then trap_move is the position of the trap cell, else trap_move = None

        Returns:
        -------
        tuple (tuple, tuple)
            return (old_position, new_position) represent a move of the AI base on the state of the board
        """

        result = self.minimax(board, player, maximizer=1, alpha=float('-inf'), beta=float('inf'), depth=self.MINIMAX_DEPTH, trap_move=trap_move)

        # Optional
        #print('Maximum value: ' + "{:.2f}".format(result[0]))
        
        if result[1] == None:
            raise "No move available for me!"
        
        return (result[1][0], result[1][1])


    def ai_move(self, board, player, trap_move = None):
        """
        Parameters:
        -----------
        board : 2D list
            the board state
        player : -1 or 1
            the type of player
        trap_move : tuple(int, int)
            if the opponent's move is a trap move, then trap_move is the position of the trap cell, else trap_move = None

        Returns
        -------
        tuple(int, int)
            if AI's move is a trap move then return the position of the trap cell, else return None 
        """
        
        start = time()
        move = self.move(board, player, trap_move)
        #print(f'AI move: {move[0]} -> {move[1]}. Time taken: {time() - start}')
        return self.move_piece(board, player, move[0], move[1])


    def player_move(self, board ,player, trap_move = None):
        """ Prompt the player for a move

        Parameters:
        -----------
        board : 2D list
            the board state
        player : -1 or 1
            the type of player
        trap_move : tuple(int, int)
            if the opponent's move is a trap move, then trap_move is the position of the trap cell, else trap_move = None

        Returns
        -------
        tuple(int, int)
            if player's move is a trap move then return the position of the trap cell, else return None 
        """

        valid_move = False
        while not valid_move:
            print(f'Player {player} choose a piece to move')
            print('Your value: ', player) 

            if trap_move == None:
                x = int(input('Enter x coordinate: '))
                y = int(input('Enter y coordinate: '))
                if board[x][y] != player:
                    print('Error: That\'s not your piece. Please choose another one')
                    continue
                print('Move your piece to?')
                new_x = int(input('Enter new x coordinate: '))
                new_y = int(input('Enter new y coordinate: '))
                if board[new_x][new_y] != 0:
                    print('Error: The position is already occupied. Please try again')
                    continue    
                avail_moves = self.get_nearby_empty_cell(board, x, y)
                if (new_x, new_y) in avail_moves:
                    valid_move = True
                    is_trap = self.move_piece(board, player, (x, y), (new_x, new_y))
                    print('OK. Valid move')
                    if is_trap != None:
                        print('Your move is a trap move')
                    return is_trap
                else:
                    print('Error: Invalid move, please try again')
            else:
                print(f'You got caught in a trap. You have to move to {trap_move}')
                x = int(input('Enter x coordinate: '))
                y = int(input('Enter y coordinate: '))
                if board[x][y] != player:
                    print('Error: That\'s not your piece. Please try again')
                    continue
                valid_move = True
                is_trap = self.move_piece(board, player, (x, y), trap_move)
                print('OK. Valid move')                    
                if is_trap != None:
                    print('Your move is a trap move')
                    return is_trap
        return None
    

    def is_in_range(self, x, y):
        """
            return True if (x, y) is a valid coordinate.
        """

        return x >= 0 and x <= 4 and y >= 0 and y <= 4


    def dfs_isolated_cells(self, board, x, y, visited : list, not_isolated : list):
        """ Find all isolated cells from a position (Used in 'vay' move)

        Parameters:
        -----------
        board : 2D list
            the board state
        x, y : int
            x, y is the coordinate of the cell
        visited : list
            the list of visited cells
        not_isolated : list
            the list of cells that is guaranteed to be not isolated

        Returns:
        --------
        bool
            return True if position (x, y) is isolated by opponent's cell 
            else return False
        """

        visited.append((x, y))
        if (x, y) in not_isolated:
            return False
        cell_type = board[x][y]
        nearby = self.get_all_nearby_cell(x, y)
        for cell in nearby:
            if cell in visited:
                continue
            if board[cell[0]][cell[1]] == 0:
                not_isolated.append((x, y))
                return False
            if board[cell[0]][cell[1]] == cell_type:
                result = self.dfs_isolated_cells(board, cell[0], cell[1], visited, not_isolated)
                if result == False:
                    not_isolated.append((x, y))
                    return False
        return True
        

    def move_piece(self, board, player, old_cell, new_cell):
        """ Move a piece from a position to a new position. Perform 'ganh', 'vay', and trap move at the same time.

        Parameters:
        -----------
        board : 2D list
            the board state
        player : 1 or -1
            the type of player
        old_cell : tuple(int, int)
            the position of the cell you want to move
        new_cell : tuple(int, int)
            the new position of that cell

        Returns:
        --------
        tuple(int, int)
            if the move is a trap move, return a tuple represents the trap position.
            else return None
        """

        dif_player = -player
        assert(board[old_cell[0]][old_cell[1]] == player and board[new_cell[0]][new_cell[1]] == 0)

        board[new_cell[0]][new_cell[1]] = player
        board[old_cell[0]][old_cell[1]] = 0
        to_be_changed = []
        diff_type_cells = self.get_nearby_cells_of_player(board, dif_player, new_cell[0], new_cell[1])
        not_isolated = []
        check_ganh = False

        for dif_cell in diff_type_cells:
            # Ganh
            ganh_opposite_x = 2 * new_cell[0] - dif_cell[0]
            ganh_opposite_y = 2 * new_cell[1] - dif_cell[1]
            if self.is_in_range(ganh_opposite_x, ganh_opposite_y):
                if board[ganh_opposite_x][ganh_opposite_y] == dif_player:
                    to_be_changed.append((ganh_opposite_x, ganh_opposite_y))
                    to_be_changed.append((dif_cell[0], dif_cell[1]))
                    # print(f'Ganh:{dif_cell}')

            # Chet
            # chet_opposite_x = 2 * dif_cell[0] - new_cell[0]
            # chet_opposite_y = 2 * dif_cell[1] - new_cell[1]
            # if self.is_in_range(chet_opposite_x, chet_opposite_y):
            #     if board[chet_opposite_x][chet_opposite_y] == cell_value:
                    # to_be_change.append((dif_cell[0], dif_cell[1]))
                    # print(f'Chet: {dif_cell}')
        for cell in to_be_changed:
            check_ganh = True
            board[cell[0]][cell[1]] = player

        # print('----------MOVE PIECE-----------')
        # self.show_board(board)
        # print('-------------------------------')

        # Vay
        to_be_changed = []
        opponent_pieces = self.get_player_pieces(board)[-player]
        for dif_cell in opponent_pieces:
            visited = []
            result = self.dfs_isolated_cells(board, dif_cell[0], dif_cell[1], visited, not_isolated)
            if result == True:
                for cell in visited:
                    to_be_changed.append(cell)
            # print('Vay:', end='')
            # print(visited)
        for cell in to_be_changed:
            check_ganh = True
            board[cell[0]][cell[1]] = player

        # Determine if the move is a trap moves

        diff_player_cells = self.get_nearby_cells_of_player(board, dif_player, old_cell[0], old_cell[1])
        # If there is no different-type cell nearby the old cell, then skip (no trap move)
        if len(diff_player_cells) != 0 and check_ganh == False:
            same_type_cells = self.get_nearby_cells_of_player(board, player, old_cell[0], old_cell[1])
            # Check if there is 2 symmetrical cell
            for cell in same_type_cells:
                trap_opposite_x = old_cell[0] * 2 - cell[0]
                trap_opposite_y = old_cell[1] * 2 - cell[1]
                if self.is_in_range(trap_opposite_x, trap_opposite_y):
                    if board[trap_opposite_x][trap_opposite_y] == player:
                        return (old_cell[0], old_cell[1])
        return None
        
        
    def minimax(self, board, player, maximizer, alpha, beta, depth, trap_move = None):
        """
        Parameters:
        -----------
        board : 2D list
            the board state
        player : 1 or -1
            the type of player
        maximizer : 0 or 1
            1: the current player is trying to maximize board's value. -1: the current player is trying to minimize the board's value
        alpha, beta : int
            alpha-beta prunning
        depth : int
            maximum depth of game tree
        trap_move: tuple(int, int)
            if the opponent's move is a trap move, then trap_move is the position of the trap cell, else trap_move = None
            
        Returns:
        --------
        (best_value, best_move, best_depth) : tuple(int, int, int)
            best_value : the best value of the board state
            best_move : the optimal move to get the most favourable board
            best_depth : how many move is needed. The higher the depth, the fewer move is required to achieve the best board
        """

        # print(f'depth:{depth}')
        # print(f'maximizer:{maximizer}')
        # print(f'board:')
        # self.show_board(board)

        if depth == 0:
            value = self.evaluate_board(board, player)
            # print(f'value:{value}')
            return (value, None, depth)

        # If we already have the winner. Stop generating new board state
        player_pieces = self.get_player_pieces(board)
        if len(player_pieces[-player]) == 0:
            value = float('inf')
            return (value, None, depth)

        if len(player_pieces[player]) == 0:
            value = float('-inf')
            return (value, None, depth)

        if maximizer == 1:
            # AI
            best_value = float('-inf')
            best_move = None
            best_depth = float('-inf')

            # If the previous opponent move was a trap move
            if trap_move != None:
                near_trap_move = self.get_nearby_cells_of_player(board, player, trap_move[0], trap_move[1])
                # For every cell near the trap cell, generate a new board state
                for cell in near_trap_move:
                    new_board = self.clone_board(board)

                    self.move_piece(new_board, player, cell, trap_move)
                    result = self.minimax(new_board, player, 0, alpha, beta, depth - 1, None)

                    if best_value < result[0] or best_value == result[0] and result[2] > best_depth:
                        best_value = result[0]
                        best_move = (cell, trap_move)
                        best_depth = result[2]

                    if result[0] >= beta:
                        return (result[0], (cell, trap_move), best_depth)

                    if alpha < result[0]:
                        alpha = result[0]

                return (best_value, best_move, best_depth)

            # Previous move was not a trap move

            # For every pieces available in the board's state
            for curr in player_pieces[player]:
                moves = self.get_nearby_empty_cell(board, curr[0], curr[1])
                # For every move a piece could perform, generate a new board state
                for next_cell in moves:
                    new_board = self.clone_board(board)
                    
                    trap_move = self.move_piece(new_board, player, curr, next_cell)
                    result = self.minimax(new_board, player, 0, alpha, beta, depth - 1, trap_move)

                    if best_value < result[0] or best_value == result[0] and result[2] > best_depth:
                        best_value = result[0]
                        best_move = (curr, next_cell)
                        best_depth = result[2]

                    if result[0] >= beta:
                        return (result[0], (curr, next_cell), best_depth)

                    if alpha < result[0]:
                        alpha = result[0]

            return (best_value, best_move, best_depth)

        elif maximizer == 0:
            # Player
            best_value = float('inf')
            best_move = None
            best_depth = float('-inf')

            # Opponent's move was trap move
            if trap_move != None:
                # For every cell near the trap cell, generate a new board state
                near_trap_move = self.get_nearby_cells_of_player(board, -player, trap_move[0], trap_move[1])
                for cell in near_trap_move:
                    new_board = self.clone_board(board)

                    self.move_piece(new_board, -player, cell, trap_move)
                    result = self.minimax(new_board, player, 1, alpha, beta, depth - 1, None)

                    if best_value > result[0] or (best_value == result[0] and result[2] > best_depth):
                            best_value = result[0]
                            best_move = (cell, trap_move)
                            best_depth = result[2]

                    if alpha >= result[0]:
                        return (result[0], (cell, trap_move), best_depth)
                    if beta > result[0]:
                        beta = result[0]

                return (best_value, best_move, best_depth)
            # Opponent's move was not a trap move

            # For every pieces available in the board's state
            for curr in player_pieces[-player]:
                moves = self.get_nearby_empty_cell(board, curr[0], curr[1])
                # For every move a piece could perform, generate a new board state
                for next_cell in moves:
                    new_board = self.clone_board(board)

                    trap_move = self.move_piece(new_board, -player, curr, next_cell)
                    result = self.minimax(new_board, player, 1, alpha, beta, depth - 1, trap_move)

                    if best_value > result[0] or (best_value == result[0] and result[2] > best_depth):
                        best_value = result[0]
                        best_move = (curr, next_cell)
                        best_depth = result[2]

                    if alpha >= result[0]:
                        return (result[0], (curr, next_cell), best_depth)
                    if beta > result[0]:
                        beta = result[0]

            return (best_value, best_move, best_depth)


    def has_ended(self, board):
        """
        Determine the winner of the board

        Returns:
        --------
        player_value : 0, 1 or -1
            If 0: no one wins the game 
            else player with player_value wins the game
        """
        player_pieces = self.get_player_pieces(board)
        if len(player_pieces[1]) == 0:
            return -1
        elif len(player_pieces[-1]) == 0:
            return 1
        else:
            return 0


if __name__ == '__main__':  

    board= [[  0,  0,  0,  0,  0],
            [  0,  0,  0,  0,  1],
            [  0, -1,  0,  0,  0],
            [  0,  0,  0,  0,  1],
            [  1,  0,  0,  0,  0]]
    ai_player = -1
    depth = 4
    solver = Minimax(depth, None)

    # P vs AI

    turn = -1
    trap_move = None
    while True:
        solver.show_board(solver.board)
        if ai_player == -turn:
            trap_move = solver.player_move(solver.board, turn, trap_move)
        else:
            trap_move = solver.ai_move(solver.board, turn, trap_move)

        winner = solver.has_ended(solver.board)
        if winner != 0:
            print(f'Player {winner} has won!!!')
            solver.show_board(solver.board)
            break
        turn = -turn

    # solver.show_board(board)
    # solver.ai_move(board, ai_player)
    # solver.show_board(board)

