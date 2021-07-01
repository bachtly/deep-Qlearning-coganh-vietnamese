from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from src.CoganhEnv import Coganh_v0
from time import time
from src import DQN
import numpy as np
import pickle


if __name__ == "__main__":
    ### Set up environment
    env = Coganh_v0()
    agent = DQN(0.9, 1, 4096, 1048576)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.training_network.add(Dense(2048, activation='relu'))
    agent.training_network.add(Dense(25*25, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error, metrics=['mse'])


    op2 = optimizers.RMSprop(learning_rate=0.00025)
    agent.target_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.target_network.add(Dense(2048, activation='relu'))
    agent.target_network.add(Dense(25*25, activation='linear'))
    agent.target_network.compile(optimizer=op2, loss=losses.mean_squared_error, metrics=['mse'])
    agent.update_target_network()

    reward_records = list()
    loss_records = list()
    count = 0
    target_update = 500
    record = 0
    max_steps = 20

    ### Training process
    total_time = 0
    # for ep in range(2):
    for ep in range(10000):
        start = time()
        _player = np.random.randint(0,2)
        _player = 2*_player-1
        player = 1
        state = env.reset(player)
        done = False
        # print(ep, '------------------', 'current epsilon: ', agent.epsilon_greedy.epsilon)
        for i in range(max_steps):
            # print(np.array(state).reshape((5,5)))
            if player == 1:
                reward, done = env.env_act()
                state = env.board.copy()
                player = -1
            
            action = agent.observe_on_training(state, action_space = env.get_act_space())
            state, reward, done, _ = env.step(action)
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
        agent.epsilon_greedy.decay(0.99997, 0.1)

        total_time += time() - start

        if ep%500 == 500-1:
            with open('cp/coganh_checkpoint_' + str(ep) + '.pkl', 'wb') as f:
                pickle.dump(agent.training_network.get_weights(), f, pickle.HIGHEST_PROTOCOL)

        if ep%10 == 9: 
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Total time after", ep, "episodes:", total_time)
            total_time = 0

        if ep%50 == 50-1:
            plt.plot(range(len(reward_records)),  reward_records)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('MSE')
            plt.savefig('performance/reward_' + str(ep) + '.png')

            loss = [(sum(loss)/len(loss))for loss in loss_records if loss != None]
            plt.plot(range(len(loss)),  loss)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('MSE')
            plt.savefig('performance/mseloss_' + str(ep) + '.png')