from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
import tensorflow as tf
from src.CoganhEnv import CoganhvsMinimax_v0
from src.DQN import DQN
import pickle
import numpy as np


with tf.device('/cpu:0'):
    max_steps = 20  
    agent = DQN(0.9, 1, 8192, 1048576)
    # agent = DQN(0.9, 1, 4096, 524288)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(2048, activation='relu'))
    agent.training_network.add(Dense(25*25, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error, metrics=['mse'])
    w = pickle.load(open('cp/hybrid/cp_5000.pkl','rb'))
    agent.training_network.set_weights(w)

    result_vs_minimax = {}
    for i in range(-32,33): result_vs_minimax[i] = 0

    env_test = CoganhvsMinimax_v0()
    for (player, ep) in [(i,j) for i in [-1,1] for j in range(1,5)]:
        done = False
        reward = 0
        depth  = ep
        state = env_test.reset(player,depth)
        print('Game start player:', player, 'depth', depth)

        for i in range(max_steps):
            # env_test.mnm_env.show_board(None)
            # print(np.array(state).reshape((5,5)))
            # print(np.array(state).reshape((5,5)))
            if player == 1:
                reward, done = env_test.env_act()
                state = env_test.board.copy()
                player = -1
                # env_test.mnm_env.show_board(None)
            
            action = agent.observe(state, action_space = env_test.get_act_space())
            state, reward, done, _ = env_test.step(action)
            # env_test.mnm_env.show_board(None)

            if done: break


        result_vs_minimax[reward] += 1
        # print_board(state)
        if reward > 0: 
            print("AI player won!")
            print(reward)
        elif reward == 0: print("AI player tie!")
        else: 
            print("AI player lost!")
            print(reward)

    # print('Result:')
    # for i in range(-32, 33):
    #     print("Point:", i, " ====>>>> ", result_vs_minimax[i])