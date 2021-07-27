from src.CoganhEnv import CoganhMvsM_v0
import tensorflow as tf
from time import time
import numpy as np


with tf.device('/cpu:0'):
    
    env = CoganhMvsM_v0()
    max_steps = 25
    result = {'won':0, 'lost':0, 'tie':0}

    start = time()
    depth1  = 1
    depth2  = 0
    print("Mvsrandom for minimax", depth1)
    for ep in range(1000):
        player = np.random.choice([-1,1], 1)[0]
        done = False
        
        state = env.reset(player,depth1, depth2)

        for i in range(max_steps*2+1):
            state, reward, done, _ = env.step()
            if done: break

        if reward > 0: 
            result['won'] += 1
            print("WON")
        elif reward < 0: 
            result['lost'] += 1
            print("LOST")
        else: 
            result['tie'] += 1
            print("TIE")

    print("Won games:", result['won'])
    print("Tie games:", result['tie'])
    print("Lost games:", result['lost'])
    print("Total time", time() - start, 's.')
