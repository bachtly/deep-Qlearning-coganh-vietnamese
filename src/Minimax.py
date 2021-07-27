from time import time
import numpy as np
import random

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

        nearby = self.get_all_nearby_cell(x, y)
        emp_cells = [ c for c in nearby if board[c[0]][c[1]] == 0] 
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
        result = [ (c[0], c[1]) for c in nearby if board[c[0]][c[1]] == player]
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

        value = np.sum(self.board)*player
        if value == 16: return 10
        elif value == -16: return -10
        return value//2

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

        new_board = [i.copy() for i in board]
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
        return self.move_piece(board, player, move[0], move[1]), move

    def aivsai_move(self, board, player, trap_move = None):
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
        return self.move_piece(board, player, move[0], move[1]), move

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
        self.board = board

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

class MinimaxInterface:
    def __init__(self, depth, save_move={}):
        self.mnm_env = Minimax(depth, None)
        self.board = self.mnm_env.board
        self.save_move = save_move
        self.depth = depth
    
    def sample_act(self, board, trap_move, turn = 1): 
        # print("sampling ... ")
        board = np.array(board).reshape((5,5))
        board = [list(line) for line in board]

        action_space = self.get_act_space(board, trap_move, turn)
        # print(board)
        # print("TRap move:", trap_move)
        # for i in action_space: 
        #     print('(',i//25//5, i//25%5,')', '(',i%25//5, i%25%5,')')

        rand_idx = random.randint(0, len(action_space)-1)
        action = action_space[rand_idx]
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5)

        trap_move = self.mnm_env.move_piece(board, turn, old, new)
        self.board = self.mnm_env.board 
        # print(self.board)

        return trap_move, action

    def minimax_act(self, board, trap_move, turn = 1): 
        # print("thinking ...")
        # print(self.board)
        if self.depth == 0: 
            return self.sample_act(board, trap_move, turn)

        env_board = np.array(board).reshape((5,5))
        env_board = [list(line) for line in env_board]
        try:
            move = self.save_move[tuple(board)]
            old, new = move
            trap_move = self.mnm_env.move_piece(env_board, turn, old, new)
            # use_dict = True
        except:
            # print("=>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print(self.mnm_env.board )
            trap_move, move = self.mnm_env.ai_move(env_board, turn, trap_move)
            # print(self.mnm_env.board )
            # use_dict = False

        self.board = self.mnm_env.board 
        old, new = move
        action = (old[0]*5+old[1])*25 + new[0]*5+new[1]
        # print(self.board)

        return trap_move, action

    def act(self, board, action, player):
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5) 
        board = np.array(board).reshape((5,5))
        board = [list(line) for line in board]

        trap_move = self.mnm_env.move_piece(board, player, old, new)
        self.board = self.mnm_env.board
        return trap_move

    def get_act_space(self, board, trap_move, player=1):
        action_space = []
        pieces = self.mnm_env.get_player_pieces(board)
        if trap_move is None:
            for old in pieces[player]:
                news = self.mnm_env.get_nearby_empty_cell(board, old[0], old[1])
                action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for new in news]
        else:
            ### old cells are those near trap position
            olds = self.mnm_env.get_nearby_cells_of_player(board, \
                                        player, trap_move[0], trap_move[1])
            new = trap_move
            action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for old in olds]
        
        return action_space

    def get_board(self):
        return [i for lst in self.board for i in lst]

    def reverse(self, board):
        board = [-i for i in board]
        board.reverse()
        return board

###================================================================================================
class Minimax:

    def __init__(self, depth, board = None):
        self.MINIMAX_DEPTH = depth
        self.WINNING = 1e6
        # self.step_remain = 30
        if board != None:
            self.board = board
        else:
            self.board = [[  1,  1,  1,  1,  1],
                          [  1,  0,  0,  0,  1],
                          [  1,  0,  0,  0, -1],
                          [ -1,  0,  0,  0, -1],
                          [ -1, -1, -1, -1, -1]]

        self.weights =   [[  1,   1,   1,   1 , 1],
                          [  1, 1.1,   1, 1.1,  1],
                          [  1,   1,   1,   1,  1],
                          [  1, 1.1,   1,1.1,  1],
                          [  1,   1,   1,   1,  1]]
        self.maximizer_pruned = 0
        self.minimizer_pruned = 0

    def shortest_path(self, board, cell_1, cell_2):
        visited = [cell_1]
        cell_queue = [(cell_1, 0)]
        while (len(cell_queue) > 0):
            (cell, length) = cell_queue.pop(0)
            movables = self.get_nearby_empty_cell(board, cell[0], cell[1])
            for movable in movables:
                if movable == cell_2:
                    return length+1
                if movable not in visited:
                    cell_queue.append((movable, length+1))
                    visited.append(movable)

        return float('inf')

    def manhattan_dist(self, cell_1, cell_2): 
        return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

    def show_board(self, board = None):
        """ Display a game board

        Parameters:
        -----------
        board : 2D list
            The board you want to display
        """

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

    def evaluate_board(self, board, player, count_empty_near_attacking = 0):
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
        if value == 16: return value
        return value 
        # return value - count_empty_near_attacking / 20

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

    def move(self, board, player, trap_move = None, remain_time = 90):

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
        search_depth = self.MINIMAX_DEPTH
        # search_depth = 0
        # if self.step_remain <= 2:
        #     search_depth = 2
        # elif remain_time >= 60:
        #     search_depth = self.MINIMAX_DEPTH
        # elif remain_time >= 15:
        #     search_depth = 6
        # elif remain_time >= 4:
        #     search_depth = 5
        # elif remain_time >= 2:
        #     search_depth = 4
        # else:
        #     search_depth = 2
        # self.step_remain -= 1
        result = self.minimax(board, player, maximizer=1, alpha=float('-inf'), beta=float('inf'), depth=search_depth, trap_move=trap_move)

        # Optional
        # print('Maximum value: ' + "{:.2f}".format(result[0]))
        # if result[0] == -self.WINNING:
        #     print('AI is definitely losing')
        # elif result[0] <= -1:
        #     print('AI may lose')
        # elif result[0] <= 0 and result[0] > -1:
        #     print('AI may draw against player')
        # elif result[0] == self.WINNING:
        #     print('AI is definitely winning')
        # else:
        #     print('AI may win')
        # print('The board AI is thinking about:')
        # self.show_board(result[2])
        return (result[1][0], result[1][1])

    def ai_move(self, board, player, trap_move = None, remain_time = 90):
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
            
        self.maximizer_pruned = 0
        self.minimizer_pruned = 0
        move = self.move(board, player, trap_move, remain_time)
        # print(f'AI move: {move[0]} -> {move[1]}')
        # print(f'Maximizer pruned {self.maximizer_pruned} times, minimizer pruned {self.minimizer_pruned} times')
        return self.move_piece(board, player, move[0], move[1]), move

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
                    print('Error: That cell is not movable')
            else:
                print(f'You got caught in a trap. You have to move to {trap_move}')
                x = int(input('Enter x coordinate: '))
                y = int(input('Enter y coordinate: '))
                if board[x][y] != player:
                    print('Error: That\'s not your piece. Please try again')
                    continue
                if trap_move not in self.get_nearby_empty_cell(board, x, y):
                    print('Error: That cell is not movable')
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
        try:
            assert(board[old_cell[0]][old_cell[1]] == player and board[new_cell[0]][new_cell[1]] == 0)
        except AssertionError:
            print('ASSERTION ERROR:')
            print(f'Current player: {player}')
            print(f'Move: {old_cell} -> {new_cell}')
            self.show_board(board)
            exit()

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
        self.board = board

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
         
    def count_empty(self, board, x, y):
        all_movable = self.get_all_nearby_cell(x, y)
        count = 0
        for cell in all_movable:
            if board[cell[0]][cell[1]] == 0:
                count += 1
        return count

    def is_isolated_empty(self, board, ally_player, x, y):
        all_nearby = self.get_all_nearby_cell(x, y)
        for cell in all_nearby:
            if board[cell[0]][cell[1]] == 0:
                return False
            elif board[cell[0]][cell[1]] == ally_player:
                return False
        return True

    def minimax(self, board, player, maximizer, alpha, beta, depth, trap_move = None, empty_cell_near_attacking = []):
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
        (best_value, best_move) : tuple(int, int, int)
            best_value : the best value of the board state
            best_move : the optimal move to get the most favourable board
        """

       

        player_pieces = self.get_player_pieces(board)
        opponent_pieces = player_pieces[-player]
        ally_pieces = player_pieces[player]

        ally_near_attacking = []
        attacking_cell = [] 

        if maximizer == 1:
            if len(opponent_pieces) == 1 and len(ally_pieces) >= 3:
                attacking_cell.append(opponent_pieces[0])
                empty_cell_near_attacking = self.get_nearby_empty_cell(board, opponent_pieces[0][0], opponent_pieces[0][1])

            elif len(opponent_pieces) > 1 and len(ally_pieces) >= 3:
                empty_cell_near_attacking = []  
                adj_cell = None
                isolated_empty = []
                visited = []
                min_nearby = 1000

                for piece in opponent_pieces:
                    if piece in visited:
                        continue
                    adj_cell = None
                    empty_cells = []
                    ally_cells = []
                    not_ok = False
                    all_nearby = self.get_all_nearby_cell(piece[0], piece[1])
                    for cell in all_nearby:
                        if board[cell[0]][cell[1]] == 0:
                            if cell in isolated_empty:
                                not_ok = True
                                break
                            if self.is_isolated_empty(board, player, cell[0], cell[1]):
                                isolated_empty.append(cell)
                                not_ok = True
                                break
                            empty_cells.append(cell)
                        if board[cell[0]][cell[1]] == -player:
                            visited.append(cell)
                            nearby_opponent = self.get_nearby_cells_of_player(board, -player, cell[0], cell[1])
                            if len(nearby_opponent) > 1:
                                visited.extend(nearby_opponent)
                                not_ok = True
                                break
                            elif adj_cell != None:
                                not_ok = True
                                break
                            else:
                                adj_cell = cell

                        elif board[cell[0]][cell[1]] == player:
                            ally_cells.append(cell)

                    if not_ok == True:
                        continue      
                    # print(piece)
                    if adj_cell != None:
                        empty_cells.extend(self.get_nearby_empty_cell(board, adj_cell[0], adj_cell[1]))
                        empty_cells = set(empty_cells)
                    # print(f'empty:{empty_cells}')
                    if len(empty_cells) == 0:
                        continue
                    if min_nearby > len(empty_cells):
                        # print(piece)
                        min_nearby = len(empty_cells)
                        if adj_cell == None:
                            attacking_cell = [piece]
                        else:
                            attacking_cell = [piece, adj_cell]
                            ally_cells.extend(self.get_nearby_cells_of_player(board, player, adj_cell[0], adj_cell[1]))
                            ally_cells = set(ally_cells)
                        empty_cell_near_attacking = empty_cells
                        ally_near_attacking = ally_cells


        # print(f'depth:{depth}')
        # print(f'maximizer:{maximizer}')
        # if maximizer == 1:
        #     print(f'empty cells:{empty_cell_near_attacking}')
        #     print(f'attacking:{attacking_cell}')
        #     print(f'ally near attacking:{ally_near_attacking}')
        # self.show_board(board)
        

        if depth == 0:
            penalty = 0
            if len(empty_cell_near_attacking) == 0:
                penalty = 10
            else:
                penalty = len(empty_cell_near_attacking)
            value = self.evaluate_board(board, player, penalty)
            # print(f'Evaluated: {value}')
            return (value, None, board)

        # If we already have the winner. Stop generating new board state
        if len(opponent_pieces) == 0:
            value = self.WINNING
            return (value, None, board)

        if len(ally_pieces) == 0:
            value = -self.WINNING
            return (value, None, board)

        if maximizer == 1:
            best_value = float('-inf')
            best_move = None
            best_board = None
            moves_memoize = []
            attacking_open_cells = []
            # If the previous opponent move was a trap move
            if trap_move != None:
                near_trap_move = self.get_nearby_cells_of_player(board, player, trap_move[0], trap_move[1])
                # For every cell near the trap cell, generate a new board state
                for cell in near_trap_move:
                    new_board = self.clone_board(board)
                    self.move_piece(new_board, player, cell, trap_move)
                    # print(f'moving after trap: {cell} -> {trap_move}')
                    result = self.minimax(new_board, player, 0, alpha, beta, depth - 1, None)

                    if best_value < result[0]:
                        best_value = result[0]
                        best_move = (cell, trap_move)
                        best_board = result[2]

                    if result[0] >= beta:
                        self.maximizer_pruned += 1
                        return (best_value, best_move, best_board)

                    if alpha < result[0]:
                        alpha = result[0]

                return (best_value, best_move, best_board)

            # Previous move was not a trap move

            # For every pieces available in the board's state
            if len(empty_cell_near_attacking) != 0: 
                empty_cell_near_attacking = sorted(empty_cell_near_attacking, key=lambda cell : self.count_empty(board, cell[0], cell[1]), reverse=True)

                pieces = ally_pieces

                # choose 2 open cells
                attacking_open_cells = empty_cell_near_attacking[:2]
                # attacking = max(empty_cell_near_attacking, key= lambda cell : self.count_empty(board, cell[0], cell[1]))

                pieces_not_near_attacking = [p for p in pieces if p not in ally_near_attacking]
                # trying to move my pieces, which is not movable from the cell being attacked, closer to the cell being attacked
                if len(pieces_not_near_attacking) != 0:
                    for attacking in attacking_open_cells:
                        move_from_cells = sorted(pieces_not_near_attacking, key=lambda p : self.shortest_path(board, p, attacking))[:2]
                        for move_from_cell in move_from_cells:
                        # move_from_cell = min(pieces, key=lambda p : self.manhattan_dist(open_cell, p))
                            empty_cells = self.get_nearby_empty_cell(board, move_from_cell[0], move_from_cell[1])
                            if len(empty_cells) == 0:
                                continue
                            move_to_cell = min(empty_cells, key=lambda c : self.shortest_path(board, c, attacking))
                            if ((move_from_cell, move_to_cell)) in moves_memoize:
                                continue
                            moves_memoize.append((move_from_cell, move_to_cell))

                            new_board = self.clone_board(board)
                            trap_move = self.move_piece(new_board, player, move_from_cell, move_to_cell)

                            # print(f'moving in attacking: {move_from_cell} -> {move_to_cell}')

                            result = self.minimax(new_board, player, 0, alpha, beta, depth-1, trap_move, attacking_open_cells)
                            if best_value < result[0]:
                                best_value = result[0]
                                best_move = (move_from_cell, move_to_cell)
                                best_board = result[2]

                            if result[0] >= beta:
                                self.maximizer_pruned += 1
                                return (best_value, best_move, best_board)

                            if alpha < result[0]:
                                alpha = result[0]
                    
            for curr in player_pieces[player]:
                moves = self.get_nearby_empty_cell(board, curr[0], curr[1])
                # For every move a piece could perform, generate a new board state
                for next_cell in moves:
                    if (curr, next_cell) in moves_memoize:
                        continue
                    new_board = self.clone_board(board)
                    trap_move = self.move_piece(new_board, player, curr, next_cell)

                    # print(f'moving random: {curr} -> {next_cell}')
                    
                    result = self.minimax(new_board, player, 0, alpha, beta, depth - 1, trap_move, attacking_open_cells)

                    if best_value < result[0]:
                        best_value = result[0]
                        best_move = (curr, next_cell)
                        best_board = result[2]

                    if result[0] >= beta:
                        self.maximizer_pruned += 1
                        return (best_value, best_move, best_board)

                    if alpha < result[0]:
                        alpha = result[0]

            return (best_value, best_move, best_board)

        elif maximizer == 0:
            # Player
            best_value = float('inf')
            best_move = None
            best_board = None
            move_memoize = []
            # Opponent's move was trap move
            if trap_move != None:
                # For every cell near the trap cell, generate a new board state
                near_trap_move = self.get_nearby_cells_of_player(board, -player, trap_move[0], trap_move[1])
                for cell in near_trap_move:
                    new_board = self.clone_board(board)
                    self.move_piece(new_board, -player, cell, trap_move)

                    result = self.minimax(new_board, player, 1, alpha, beta, depth - 1, None)

                    if best_value > result[0]:
                            best_value = result[0]
                            best_move = (cell, trap_move)
                            best_board = result[2]

                    if alpha >= result[0]:
                        self.minimizer_pruned += 1
                        return (best_value, best_move, best_board)

                    if beta > result[0]:
                        beta = result[0]

                return (best_value, best_move, best_board)
            # Opponent's move was not a trap move
            
            if len(empty_cell_near_attacking) != 0:
                for cell in empty_cell_near_attacking:
                    if board[cell[0]][cell[1]] != 0:
                        continue
                    opponents_near_open = self.get_nearby_cells_of_player(board, -player, cell[0], cell[1])
                    for opponent in opponents_near_open:
                        move_memoize.append((opponent, cell))
                        new_board = self.clone_board(board)
                        trap_move = self.move_piece(new_board, -player, opponent, cell)
                            
                        result = self.minimax(new_board, player, 1, alpha, beta, depth - 1, trap_move)

                        if best_value > result[0]:
                            best_value = result[0]
                            best_move = (opponent, cell)
                            best_board = result[2]

                        if alpha >= result[0]:
                            self.minimizer_pruned += 1
                            return (best_value, best_move, best_board)
                        if beta > result[0]:
                            beta = result[0]
                
            # For every pieces available in the board's state
            for curr in player_pieces[-player]:
                moves = self.get_nearby_empty_cell(board, curr[0], curr[1])
                # For every move a piece could perform, generate a new board state
                for next_cell in moves:
                    if (curr, next_cell) in move_memoize:
                        continue
                    new_board = self.clone_board(board)
                    trap_move = self.move_piece(new_board, -player, curr, next_cell)
                        
                    result = self.minimax(new_board, player, 1, alpha, beta, depth - 1, trap_move)

                    if best_value > result[0]:
                        best_value = result[0]
                        best_move = (curr, next_cell)
                        best_board = result[2]

                    if alpha >= result[0]:
                        self.minimizer_pruned += 1
                        return (best_value, best_move, best_board)
                    if beta > result[0]:
                        beta = result[0]

            return (best_value, best_move, best_board)
 
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



