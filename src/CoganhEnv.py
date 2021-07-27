from numpy.lib.twodim_base import _trilu_dispatcher
from src.EpsilonGreedy import EpsilonGreedy, EpsilonZero
from src.Minimax import Minimax, MinimaxInterface
import numpy as np
import random

class Coganh_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_move = {}):
        self.board = None
        self.mnm_env = MinimaxInterface(4, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1
        self.save_move = save_move

    def reset(self, player, depth=4, save_move = {}):
        self.mnm_env = MinimaxInterface(depth, save_move)
        self.board = self.mnm_env.get_board()
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 1, True
        elif sum_point == 16:
            return -1, True
        if sum_point <= -8:
            return 1, False
        elif sum_point >= 8:
            return -1, False

        return 0, False

    def env_act(self):
        self.trap_move, _ = self.mnm_env.minimax_act(self.board, self.trap_move)
        # try:
        #     move = self.save_move[tuple(self.board)]
        #     old, new = move
        #     self.trap_move = self.mnm_env.move_piece(self.mnm_env.board, \
        #                                         self.current_turn, old, new)
        #     use_dict = True
        # except:
        #     self.trap_move, _ = self.mnm_env.ai_move(self.mnm_env.board, \
        #                                         self.current_turn, self.trap_move)
        #     use_dict = False

        self.board = self.mnm_env.get_board()
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        return reward, done, _

    def step(self, action):
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5)
        if not self.is_valid(action):
            raise Exception('Invalid action')

        self.trap_move = self.mnm_env.act(self.board, action, self.current_turn)
        self.board = self.mnm_env.get_board()
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1

        if done:
            return self.board.copy(), reward, done, None

        reward, done, use_dict = self.env_act()
        return self.board.copy(), reward, done, use_dict

    def get_act_space(self):
        if self.current_turn == 1 and self.trap_move is not None: 
            self.trap_move = (4-self.trap_move[0], 4-self.trap_move[1])
        
        if self.current_turn == 1: 
            board = self.mnm_env.reverse(self.board)
        else:
            board = self.board
        board = [list(i) for i in np.array(board).reshape((5,5))]
        ret = self.mnm_env.get_act_space(board, self.trap_move, -1)
        return ret

    def is_valid(self, action):
        old, new = action//25, action%25
        if self.current_turn == self.board[old]:
            if self.board[new] == 0: return True
        return False

class CoganhvsMinimax_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_move={}):
        self.board = None
        self.mnm_env = Minimax(4, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1
        self.save_move = save_move

    def reset(self, player, depth=4):
        self.mnm_env = Minimax(depth, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 16, True
        elif sum_point == 16:
            return -16, True
        return -np.sum(self.board), False

    def env_act(self):
        # print("==============================================================")
        # print(self.mnm_env.board)
        self.trap_move, _ = self.mnm_env.ai_move(self.mnm_env.board, \
                                                self.current_turn, self.trap_move)
        # print(self.mnm_env.board)
        # print("==============================================================")
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

class CoganhvsHuman_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self):
        self.board = None
        self.mnm_env = Minimax(4, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1

    def reset(self, player, depth=4):
        self.mnm_env = Minimax(depth, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 16, True
        elif sum_point == 16:
            return -16, True
        return -np.sum(self.board), False

    def env_act(self):
        self.mnm_env.show_board(self.mnm_env.board)
        input_board = [list(i) for i in np.array(self.board).reshape((5,5))]
        self.trap_move = self.mnm_env.player_move(input_board, \
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

        input_board = [list(i) for i in np.array(self.board).reshape((5,5))]
        self.trap_move = self.mnm_env.move_piece(input_board, \
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

class CoganhMvsM_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self):
        self.depth = {-1: 0, 1: 0}
        self.player_mark = -1
        self.current_turn = -1

        self.trap_move = None

        self.mnm_env = None
        self.board = None

    def reset(self, player, depth1=4, depth2=5):
        self.depth = {-1: depth1, 1: depth2}
        self.current_turn = player

        self.trap_move = None
        self.board = MinimaxInterface(4, {}).get_board()
        return self.board.copy()

    def check_win(self):
        # print("===================================>")
        # print(self.board)
        # print(-np.sum(self.board))
        sum_point = -np.sum(self.board)
        if sum_point == 16:      
            return 16, True
        elif sum_point == -16:
            return -16, True
        return sum_point, False

    def env_act(self):
        return 0

    def step(self):
        turn = self.current_turn
        self.mnm_env = MinimaxInterface(self.depth[turn], {})
        # print("=====================================================>")
        # print("TURN, DEPTH", turn, self.depth[turn])
        # print(np.array(self.board).reshape((5,5)))
        self.trap_move, _ = self.mnm_env.minimax_act(self.board.copy(), self.trap_move, turn)
        # print(np.array(self.mnm_env.board))
        # print("=====================================================>")
        self.board = self.mnm_env.get_board()
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1

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

class CoganhMovegen_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_dict = {}):
        self.board = None
        self.mnm_env = Minimax(4, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1
        self.save_dict = save_dict

    def reset(self, player, depth=4):
        self.mnm_env = Minimax(depth, None)
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
        try:
            old, new = self.save_dict[tuple(self.board)]
            self.trap_move = self.mnm_env.move_piece(self.mnm_env.board, \
                                                self.current_turn, old, new)
            use_dict = True
        except:
            self.trap_move, move = self.mnm_env.aivsai_move(self.mnm_env.board, \
                                                self.current_turn, self.trap_move)
            self.save_dict[tuple(self.board)] = move
            use_dict = False
      
        self.board = [i for lst in self.mnm_env.board for i in lst]
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        return reward, done, use_dict

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

        reward, done, use_dict = self.env_act()
        return self.board.copy(), reward, done, use_dict

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

class CoganhvsRandom_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_move={}):
        self.board = None
        self.mnm_env = Minimax(4, None)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1
        self.save_move = save_move

    def reset(self, player, depth=4):
        self.mnm_env = Minimax(depth, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = -np.sum(self.board)
        if sum_point == 16:      
            return 16, True
        elif sum_point == -16:
            return -16, True
        return sum_point, False

    def env_act(self):
        action_space = self.get_act_space()
        random_idx = random.randint(0, len(action_space)-1)
        action = action_space[random_idx]
        old, new = action//25, action%25
        old, new = (old//5, old%5), (new//5, new%5)
        if not self.is_valid(action):
            raise Exception('Invalid action')

        self.trap_move = self.mnm_env.move_piece(self.mnm_env.board, \
                                                self.current_turn, old, new)
    
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

class CoganhZero_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_move = {}):
        self.board = None
        self.trap_move = None

        self.player_mark = -1
        self.current_turn = -1
        self.save_move = save_move

        self.depth = 4
        self.depth2 = 0

        self.mnm_env = MinimaxInterface(4)
        self.epsilon = EpsilonZero(0)

    def reset(self, player, depth=4, depth2=0):
        self.mnm_env = MinimaxInterface(depth, self.save_move)
        self.board = self.mnm_env.get_board()
        self.current_turn = player
        self.player_mark = player   

        self.depth = depth
        self.depth2 = depth2

        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = -np.sum(self.board)
        if sum_point in [-16, 16]:
            return sum_point//16, True
        elif -16 < sum_point <= -6:
            return -1, False
        elif 6 <= sum_point < 16:
            return 1, False
        return 0, False

    def play(self, max_steps, player):
        done = False
        state_lst, reward_lst, action_lst = [], [], []
        reward = 0
        random_factor = np.random.choice([-1,1], 1)[0]
        for z in range(max_steps*2):
            env_1 = MinimaxInterface(self.depth, self.save_move)
            env_2 = MinimaxInterface(self.depth2, self.save_move)

            if player == -1: 
                state_lst += [self.board.copy()]
                reward_lst += [reward]
            else: 
                state_lst += [self.mnm_env.reverse(self.board)]
                reward_lst += [-reward]

            if player == -1: self.board = self.mnm_env.reverse(self.board)
            
            # ret = self.epsilon.perform()
            ret = random_factor==player
            if self.trap_move is not None: self.trap_move = (4-self.trap_move[0], 4-self.trap_move[1])

            if ret: 
                self.trap_move, action = env_2.minimax_act(self.board, self.trap_move)
                self.board = env_2.get_board()
            else: 
                self.trap_move, action = env_1.minimax_act(self.board, self.trap_move)
                self.board = env_1.get_board()

            if player == -1: self.board = self.mnm_env.reverse(self.board)

            reward, done = self.check_win()

            action_lst += [624-action]

            player = -player
            if done: break

        # for i in range(len(state_lst)):
        #     print(np.array(state_lst[i]).reshape((5,5)))
        #     print(reward_lst[i])
        #     a = action_lst[i]
        #     print(a//25//5, ',', a//25%5, '->', a%25//5, ',', a%25%5)
        # if reward_lst[-1]==0:
        #     [print(np.array(i).reshape((5,5))) for i in state_lst]
        #     exit()
        # exit()

        return state_lst, reward_lst, action_lst, done

    def get_act_space(self):
        self.mnm_env.get_act_space(self.board, self.trap_move)

    def is_valid(self, action):
        old, new = action//25, action%25
        if self.current_turn == self.board[old]:
            if self.board[new] == 0: return True
        return False

class CoganhAIvAI_v0:
    ### in this game, -1 is the maximizer player
    def __init__(self, save_move = {}):
        self.board = None
        self.mnm_env = MinimaxInterface(4)
        self.trap_move = None
        self.current_turn = -1
        self.player_mark = -1
        self.save_move = save_move

    def reset(self, player, depth=4):
        self.mnm_env = MinimaxInterface(depth, None)
        self.board = self.mnm_env.board
        self.board = [i for lst in self.board for i in lst]
        self.current_turn = player
        self.player_mark = -1   
        self.trap_move = None
        return self.board.copy()

    def check_win(self):
        sum_point = np.sum(self.board)
        if sum_point == -16:      
            return 10, True
        elif sum_point == 16:
            return -10, True
        return -sum_point/2, False

    def step(self, action):
        # print("==================================================================")
        if self.current_turn == 1: self.board = self.mnm_env.reverse(self.board)
        # print(np.array(self.board).reshape((5,5)))
        self.trap_move = self.mnm_env.act(self.board, action, -1)
        self.board = self.mnm_env.get_board()
        if self.current_turn == 1: self.board = self.mnm_env.reverse(self.board)
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        # print(np.array(self.board).reshape((5,5)))
        # print("==================================================================")
        return self.board.copy(), reward, done

    def get_act_space(self):
        if self.current_turn == 1 and self.trap_move is not None: 
            self.trap_move = (4-self.trap_move[0], 4-self.trap_move[1])
        
        if self.current_turn == 1: 
            board = self.mnm_env.reverse(self.board)
        else:
            board = self.board
        board = [list(i) for i in np.array(board).reshape((5,5))]
        ret = self.mnm_env.get_act_space(board, self.trap_move, -1)
        return ret

    def is_valid(self, action):
        old, new = action//25, action%25
        if self.current_turn == self.board[old]:
            if self.board[new] == 0: return True
        return False

