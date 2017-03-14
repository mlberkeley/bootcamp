import gym
import numpy as np
import time
import _pickle

env = gym.make('FrozenLake-v0')
with open('Q_table.pkl', 'rb') as f:
    Q = _pickle.load(f)

score = 0
for i in range(100):
    done = False
    state = env.reset()
    while not done:
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        reward = reward * 100
        if i > 97:
            env.render()
            time.sleep(1)
        if reward == 100:
            score += 1
print(score)
