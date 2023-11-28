import gymnasium as gym
import numpy as np
from collections import defaultdict
env = gym.make('Blackjack-v1', natural=False, sab=False)
#env.metadata['render_fps'] = .5

Q = defaultdict(lambda: np.zeros(env.action_space.n))

num=10000
epsilon = 1

for i in range(num):
    state, _ = env.reset()
    done = False

    while not done: 
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        Q[state][action] = Q[state][action] + .01 * reward + .95
        
        state = next_state
        done = terminated or truncated
    if epsilon > .1:
        epsilon = epsilon/2
