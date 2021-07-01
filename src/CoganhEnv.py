import Minimax
import numpy as np

class Coganh_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self):
        self.board = None
        self.mnm_env = Minimax(3, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1

    def reset(self, player):
        self.mnm_env = Minimax(3, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 32, True
        elif sum_point == 16:
            return -32, True
        return -np.sum(self.board), False

    def env_act(self):

        self.trap_move = self.mnm_env.ai_move(self.mnm_env.board, \
                                                self.current_turn, self.trap_move)
        self.board = [i for lst in self.mnm_env.board for i in lst]
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        return reward, done

    def step(self, action):
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5)
        if not self.is_valid(action):
            raise Exception('Invalid action')

        self.trap_move = self.mnm_env.move_piece(self.mnm_env.board, \
                                                self.current_turn, old, new)
        self.board = [i for lst in self.mnm_env.board for i in lst]
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1

        if done:
            return self.board.copy(), reward, done, None

        reward, done = self.env_act()
        return self.board.copy(), reward, done, None

    def get_act_space(self):
        action_space = []
        pieces = self.mnm_env.get_player_pieces(self.mnm_env.board)
        if self.trap_move is None:
            for old in pieces[self.current_turn]:
                news = self.mnm_env.get_nearby_empty_cell(self.mnm_env.board, old[0], old[1])
                action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for new in news]
        else:
            ### old cells are those near trap position
            olds = self.mnm_env.get_nearby_cells_of_player(self.mnm_env.board, \
                                        self.current_turn, self.trap_move[0], self.trap_move[1])
            new = self.trap_move
            action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for old in olds]
        
        return action_space

    def is_valid(self, action):
        old, new = action//25, action%25
        if self.current_turn == self.board[old]:
            if self.board[new] == 0: return True
        return False

class CoganhvsMinimax_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self):
        self.board = None
        self.mnm_env = Minimax(3, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1

    def reset(self, player):
        self.mnm_env = Minimax(4, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 32, True
        elif sum_point == 16:
            return -32, True
        return -np.sum(self.board), False

    def env_act(self):

        self.trap_move = self.mnm_env.ai_move(self.mnm_env.board, \
                                                self.current_turn, self.trap_move)
        self.board = [i for lst in self.mnm_env.board for i in lst]
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        return reward, done

    def step(self, action):
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5)
        if not self.is_valid(action):
            raise Exception('Invalid action')

        self.trap_move = self.mnm_env.move_piece(self.mnm_env.board, \
                                                self.current_turn, old, new)
        self.board = [i for lst in self.mnm_env.board for i in lst]
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1

        if done:
            return self.board.copy(), reward, done, None

        reward, done = self.env_act()
        return self.board.copy(), reward, done, None

    def get_act_space(self):
        action_space = []
        pieces = self.mnm_env.get_player_pieces(self.mnm_env.board)
        if self.trap_move is None:
            for old in pieces[self.current_turn]:
                news = self.mnm_env.get_nearby_empty_cell(self.mnm_env.board, old[0], old[1])
                action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for new in news]
        else:
            ### old cells are those near trap position
            olds = self.mnm_env.get_nearby_cells_of_player(self.mnm_env.board, \
                                        self.current_turn, self.trap_move[0], self.trap_move[1])
            new = self.trap_move
            action_space += [ ((old[0]*5 + old[1])*25 + new[0]*5 + new[1]) for old in olds]
        
        return action_space

    def is_valid(self, action):
        old, new = action//25, action%25
        if self.current_turn == self.board[old]:
            if self.board[new] == 0: return True
        return False
