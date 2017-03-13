import gym
import numpy as np
env = gym.make('FrozenLake-v0')
# grids
Q = np.zeros((env.observation_space.n, env.action_space.n))
# learning parameters

lr = 0.85
y = 0.99

num_steps = 2000

rList = []
for ii in range(num_steps):
    j = 0
    # reset environment, get first observation
    s = env.reset()
    rAll = 0
    d = False
    while j < 99:
        j += 1
        # choose action greedily
        a  = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*1.0/ (ii+1))
        s1, r, d, _ = env.step(a)
        r = r * 100
        # update Q table
        Q[s, a] = Q[s,a] + lr * (r + y * np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:break
    rList.append(rAll)
    env.render()
print("Score over time: " +  str(sum(rList)/num_steps))
print("Final Q-Table Values")
print(Q)
