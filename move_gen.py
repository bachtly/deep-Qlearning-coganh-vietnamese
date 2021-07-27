from src.CoganhEnv import CoganhMovegen_v0
from time import time
import random
import numpy as np
import pickle

### main idea is to save "clever moves" of minimax AI 
### to a dict of (tuple(state) => move)
### in order to save time to make move by DFS

if __name__ == "__main__":
    ### Set up environment
    save_move = pickle.load(open('minimax_cp/minimax_6/minimax6_checkpoint_23999.pkl', 'rb'))
    # save_move = {}
    env = CoganhMovegen_v0(save_move)
    
    reward_records = list()
    count = 0
    max_steps = 30

    ### Training process
    total_time = 0
    use_dmove = 0
    not_use_dmove = 0
    time_randmove = 0
    time_aimove = 0
    for ep in range(24000, 25000):
        start = time()
        _player = np.random.randint(0,2)
        _player = 2*_player-1
        player = -1
        state = env.reset(player,6)
        done = False
        # print(ep, '------------------', 'current epsilon: ', agent.epsilon_greedy.epsilon)'
        
        for i in range(max_steps):
            # print(np.array(state).reshape((5,5)))
            if player == 1:
                reward, done, _ = env.env_act()
                state = env.board.copy()
                player = -1
            
            newstart = time()
            action_space = env.get_act_space()
            random_idx = random.randint(0, len(action_space)-1)
            action = action_space[random_idx]
            time_randmove += time() - newstart
            newstart = time()
            state, reward, done, use_dict = env.step(action)
            time_aimove += time() - newstart

            if use_dict is True:
                use_dmove += 1
            elif use_dict is False:
                not_use_dmove += 1                

            count += 1
            if done: break

        total_time += time() - start

        if ep%500 == 500-1:
            with open('minimax_cp/minimax_6/minimax6_checkpoint_' + str(ep) + '.pkl', 'wb') as f:
                pickle.dump(env.save_dict, f, pickle.HIGHEST_PROTOCOL)

        if ep%100 == 100-1: 
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Total time after", ep, "episodes:", total_time)
            print("Not used: ", not_use_dmove)
            print("Used: ", use_dmove)
            print("Random move time: ", time_randmove)
            print("AI move time: ", time_aimove)
            use_dmove = 0
            not_use_dmove = 0
            time_randmove = 0
            time_aimove = 0
            total_time = 0
