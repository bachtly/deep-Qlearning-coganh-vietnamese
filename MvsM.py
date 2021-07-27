from src.CoganhEnv import CoganhMvsM_v0
import tensorflow as tf
import numpy as np


with tf.device('/cpu:0'):
    
    ### depth 0 is always first to move
    env_test = CoganhMvsM_v0()
    max_steps = 25

    for ep in range(1):
        player = -1
        done = False
        depth1  = 3
        depth2  = 2
        state = env_test.reset(player,depth1, depth2)
        print('game ' + str(ep) + ' start', depth1, player,'player first and play vs Minimax depth = ', depth2)

        for i in range(max_steps*2+1):
            state, reward, done, _ = env_test.step()
            # print(np.array(env_test.board).reshape((5,5)))
            # print("REWARD", reward)
            if done: break


        # print_board(state)
        print('game ' + str(ep) + ' end ---------------------', reward)

    # print('Result:')
    # for i in range(-32, 33):
    #     print("Point:", i, " ====>>>> ", result_vs_minimax[i])