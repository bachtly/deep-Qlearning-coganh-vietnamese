from tensorflow.keras import losses, optimizers
from src.CoganhEnv import CoganhvsRandom_v0
from tensorflow.keras.layers import Dense
import tensorflow as tf
from src.DQN import DQN
from time import time
import numpy as np
import pickle


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

    cpt_prefix = 'cp/cp_'
    cpt_postfix = '.pkl'
    record_results = {}
    for cpt_num in range(0, 11001, 200):
        w = pickle.load(open(cpt_prefix + str(cpt_num) + cpt_postfix,'rb'))
        agent.training_network.set_weights(w)

        result_vs_random = {'won':0, 'lost':0, 'tie':0}

        env_test = CoganhvsRandom_v0()
        start = time()
        for ep in range(1000):
            player = np.random.choice([-1,1], 1)[0]
            done = False
            reward = 0
            state = env_test.reset(player)

            for i in range(max_steps):
                if player == 1:
                    reward, done = env_test.env_act()
                    state = env_test.board.copy()
                    player = -1
                
                action = agent.observe(state, action_space = env_test.get_act_space())
                state, reward, done, _ = env_test.step(action)

                if done: break

            if reward > 0: 
                result_vs_random['won'] += 1
            elif reward < 0: 
                result_vs_random['lost'] += 1
            else: 
                result_vs_random['tie'] += 1
      
        print("Checkpoint:", cpt_num)
        print("Won games:", result_vs_random['won'])
        print("Tie games:", result_vs_random['tie'])
        print("Lost games:", result_vs_random['lost'])
        record_results[str(cpt_num)] = result_vs_random
        print("Total time", time() - start, 's.')
        if cpt_num%1000 == 0:
            pickle.dump(record_results, open('performance/WRRD/zero.pkl','wb'))

    pickle.dump(record_results, open('performance/WRRD/zero.pkl','wb'))
