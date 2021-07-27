from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
from src.CoganhEnv import CoganhvsHuman_v0
import tensorflow as tf
from src.DQN import DQN
import pickle


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

    env = CoganhvsHuman_v0()
    for ep in range(1):
        player = -1
        done = False
        reward = 0
        depth  = 4
        state = env.reset(player,depth)
        print('Linku starto!')

        for i in range(max_steps):
            if player == 1:
                reward, done = env.env_act()
                state = env.board.copy()
                player = -1
                env.mnm_env.show_board(env.mnm_env.board)
            
            action = agent.observe(state, action_space = env.get_act_space())
            state, reward, done, _ = env.step(action)
            env.mnm_env.show_board(env.mnm_env.board)

            if done: break

        print('Logu outo')
