from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
import tensorflow as tf
from src.CoganhEnv import CoganhvsRandom_v0
from src.DQN import DQN
import pickle
import numpy as np
from time import time


with tf.device('/cpu:0'):
    max_steps = 25
    agent = DQN(0.9, 1, 8192, 1048576)
    # agent = DQN(0.9, 1, 4096, 524288)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(2048, activation='relu'))
    agent.training_network.add(Dense(25*25, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error, metrics=['mse'])
    w = pickle.load(open('cp/zero4/cp_7400.pkl','rb'))
    # w = pickle.load(open('cp/coganh_totalcp_24999.pkl','rb'))
    agent.training_network.set_weights(w)

    result_vs_minimax = {'won':0, 'lost':0, 'tie':0}

    env_test = CoganhvsRandom_v0()
    start = time()
    for ep in range(100):
        _player = np.random.choice([-1,1], 1)[0]
        player = _player
        done = False
        reward = 0
        state = env_test.reset(player)
        # print('game ' + str(ep) + ' start', player,'player first and play vs Random')

        for i in range(max_steps):
            # env_test.mnm_env.show_board(None)
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

        if reward > 0: 
            result_vs_minimax['won'] += 1
            print("WON")
        elif reward < 0: 
            result_vs_minimax['lost'] += 1
            print("LOST")
        else: 
            result_vs_minimax['tie'] += 1
            print("TIE")

    
    print("Won games:", result_vs_minimax['won'])
    print("Tie games:", result_vs_minimax['tie'])
    print("Lost games:", result_vs_minimax['lost'])
    print("Total time", time() - start, 's.')

    # print('Result:')
    # for i in range(-32, 33):
    #     print("Point:", i, " ====>>>> ", result_vs_minimax[i])