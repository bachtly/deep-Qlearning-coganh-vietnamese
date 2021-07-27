from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
import tensorflow as tf
from src.CoganhEnv import CoganhAIvAI_v0
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
    w = pickle.load(open('cp/cp_11000.pkl','rb'))
    agent.training_network.set_weights(w)

    env = CoganhAIvAI_v0()
    print("Linku starto")
    for ep in range(10):
        player = 1
        done = False
        reward = 0
        depth  = 3
        state = env.reset(player,depth)
        for i in range(max_steps):
            if player == 1: state = env.mnm_env.reverse(state)

            action = agent.observe(state, action_space=env.get_act_space())
            state, reward, done = env.step(action)

            player = -player
            if done: break
        print("Reward ========> ", reward)

            
