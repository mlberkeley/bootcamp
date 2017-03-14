import gym
import numpy as np
import time
import _pickle
env = gym.make('FrozenLake-v0')
# grids
Q = np.zeros((env.observation_space.n, env.action_space.n))
# learning parameters

learning_rate = 0.99
gamma = 0.99

num_steps = 200000

rList = []
for t in range(num_steps):
    j = 0
    # reset environment, get first observation
    state = env.reset()
    rAll = 0
    d = False
    while j < 99:
        j += 1
        # choose action greedily
        action  = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)*1.0/ (t+1))
        new_state, r, done, _ = env.step(action)
        r = (r * 100)
        # update Q table
        Q[state, action] = Q[state,action] + learning_rate * (r + gamma *
                                                    np.max(Q[new_state,:])
                                                    - Q[state,action])
        rAll += r
        state = new_state
        if done:
            break
        #if t == num_steps - 1:
        #    env.render()
        #    time.sleep(1)
    rList.append(rAll)
    print("Time step: {}\r".format(t), end="")
print("Score over time: " +  str(sum(rList)/num_steps))
print("Final Q-Table Values")
print(Q)
with open("Q_table.pkl", 'wb') as f:
    _pickle.dump(Q, f)
