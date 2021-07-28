from src.CoganhEnv import CoganhMvsM_v0
import tensorflow as tf

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
        
        print('Linku starto')

        for i in range(max_steps*2+1):
            state, reward, done, _ = env_test.step()
            if done: break

        if reward > 0:
            print("Minimax", depth1, "WON!")
        elif reward < 0:
            print("Minimax", depth2, "WON!")
        else:
            print("Game tie!")
        print(reward)

        print('Logu outo')