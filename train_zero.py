from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from src.CoganhEnv import CoganhZero_v0
from src.DQN import DQNZero
from time import time
import numpy as np
import pickle


if __name__ == "__main__":
    ### Set up environment
    env = CoganhZero_v0({})
    agent = DQNZero(0.95, 1, 8192, 1048576)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(2048, activation='relu', input_shape=(25,)))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(4096, activation='relu'))
    agent.training_network.add(Dense(2048, activation='relu'))
    agent.training_network.add(Dense(25*25, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error, metrics=['mse'])


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
    target_update = 500
    epoch_per_eps = 25
    max_steps = 25
    record = 0
    count = 0

    ### Training process
    total_time = 0
    for ep in range(11000):
        start = time()

        player = np.random.choice([-1,1], 1)[0]
        state = env.reset(player,3)
        done = False
        
        ### main steps
        state_lst, reward_lst, action_lst, done = env.play(max_steps, player)
        agent.observe_on_training(state_lst, reward_lst, action_lst, done)

        ### training
        hist = [agent.train_network(64 ,64,1,verbose=0, cer_mode=True) for i in range(epoch_per_eps)]
        if hist is not None: loss_records += hist

        count += len(state_lst)
        reward_records.append(reward_lst[-1])
        
        ### update target network
        if ep%target_update == 0: 
            agent.update_target_network()

        total_time += time() - start

        ### checkpointing
        if ep%200 == 0:
            with open('cp/cp_' + str(ep) + '.pkl', 'wb') as f:
                pickle.dump(agent.training_network.get_weights(), f, pickle.HIGHEST_PROTOCOL)

        ### log time
        if ep%100 == 0: 
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Total time after", ep, "episodes:", total_time)
            total_time = 0

        ### save performance plotings
        if ep%200 == 0:
            plt.plot(range(len(reward_records)),  reward_records)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('Reward')
            plt.savefig('performance/reward_' + str(ep) + '.png')
            plt.close()

            loss = [(sum(loss)/len(loss))for loss in loss_records if loss != None]
            plt.plot(range(len(loss)),  loss)
            plt.title('Checkpoint: ' + str(ep))
            plt.xlabel('Training steps')
            plt.ylabel('MSE')
            plt.savefig('performance/mse_' + str(ep) + '.png')
            plt.close()
        
            pickle.dump(loss_records, open('performance/mse_'+str(ep)+'.pkl', 'wb'))
            pickle.dump(reward_records, open('performance/reward_'+str(ep)+'.pkl', 'wb'))