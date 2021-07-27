from numpy.lib.npyio import save
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from src.CoganhEnv import Coganh_v0
from src.DQN import DQN
from time import time
import numpy as np
import pickle


if __name__ == "__main__":
    ### Set up environment
    save_move = pickle.load(open('minimax_cp/minimax_total.pkl', 'rb'))
    # save_move = {}
    env = Coganh_v0(save_move)
    agent = DQN(0.96, 1, 8192, 1048576)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(2048, activation='relu'))
    agent.training_network.add(Dense(25*25, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error, metrics=['mse'])
    # weights = pickle.load(open('cp/coganh_totalcp_11499.pkl', 'rb'))
    # agent.training_network.set_weights(weights)

    op2 = optimizers.RMSprop(learning_rate=0.00025)
    agent.target_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.target_network.add(Dense(4096, activation='relu'))
    agent.target_network.add(Dense(4096, activation='relu'))
    agent.target_network.add(Dense(2048, activation='relu'))
    agent.target_network.add(Dense(25*25, activation='linear'))
    agent.target_network.compile(optimizer=op2, loss=losses.mean_squared_error, metrics=['mse'])
    agent.update_target_network()

    reward_records = list()
    loss_records = list()
    # loss_records = pickle.load(open('performance/mse_11499.pkl', 'rb'))
    # reward_records = pickle.load(open('performance/reward_11499.pkl', 'rb'))
    count = 0
    target_update = 500
    record = 0
    max_steps = 30

    ### Training process
    total_time = 0
    n_use_dict = 0
    n_not_use_dict = 0
    # for ep in range(2):
    for ep in range(0, 25000):
        start = time()
        _player = np.random.randint(0,1)
        _player = 2*_player-1
        player = _player

        state = env.reset(player, 0, {})
        # if ep < 3000: state = env.reset(player, 0, {})
        # elif 3000 <= ep < 4000: state = env.reset(player, 1, {})
        # elif 4000 <= ep < 5000: state = env.reset(player, 2, {})
        # elif 5000 <= ep < 6000: state = env.reset(player, 3, {})
        # elif 6000 <= ep < 7000: state = env.reset(player, 4, {})
        # elif 7000 <= ep < 8000: state = env.reset(player, 5, save_move)

        done = False
        
        # print(ep, '------------------', 'current epsilon: ', agent.epsilon_greedy.epsilon)
        for i in range(max_steps):
            # print(np.array(state).reshape((5,5)))
            if player == 1:
                reward, done, _= env.env_act()
                state = env.board.copy()
                player = -1
            
            action = agent.observe_on_training(state, action_space = env.get_act_space())
            state, reward, done, use_dict = env.step(action)

            # if use_dict is True: n_use_dict += 1
            # elif use_dict is False: n_not_use_dict += 1 

            # print(state, done)
            record += reward
            # print(ep, '-----------------------------------', reward)
            agent.take_reward(reward, state, done)
            hist = agent.train_network(64 ,64,1,verbose=0, cer_mode=True)
            loss_records.append(hist)
            count += 1
            if count % target_update == 0:
                agent.update_target_network()
                print("Target network updated.")

            if done: break

        reward_records.append(record)
        agent.epsilon_greedy.decay(0.99998, 0.1)

        total_time += time() - start

        if ep%500 == 500-1:
            with open('cp/mnm/cp_' + str(ep) + '.pkl', 'wb') as f:
                pickle.dump(agent.training_network.get_weights(), f, pickle.HIGHEST_PROTOCOL)

        if ep%500 == 500-1: 
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Total time after", ep, "episodes:", total_time)
            # print(n_not_use_dict)
            # print(n_use_dict)
            # n_use_dict = 0
            # n_not_use_dict = 0
            total_time = 0

        if ep%500 == 500-1:
            plt.plot(range(len(reward_records)),  reward_records)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('MSE')
            plt.savefig('performance/mnm/reward_' + str(ep) + '.png')
            plt.close()

            loss = [(sum(loss)/len(loss))for loss in loss_records if loss != None]
            plt.plot(range(len(loss)),  loss)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('MSE')
            plt.savefig('performance/mnm/mse_' + str(ep) + '.png')
            plt.close()

            pickle.dump(loss_records, open('performance/mnm/mse_'+str(ep)+'.pkl', 'wb'))
            pickle.dump(reward_records, open('performance/mnm/reward_'+str(ep)+'.pkl', 'wb'))