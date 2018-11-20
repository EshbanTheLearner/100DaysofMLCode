import gym
from gym import wrappers
import numpy as np

env = gym.make("FrozenLake-v0")
env = wrappers.Monitor(env, "./results", force=True)

Q = np.zeros([env.observation_space.n, env.action_space.n])
n_s_a = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 1000000
epsilon = 0.2
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    results_list = []
    result_sum = 0.0
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        results_list.append((state, action))
        result_sum += reward
        state = new_state
        rAll += reward
    rList.append(rAll)

    for (state, action) in results_list:
        n_s_a[state, action] += 1.0
        alpha = 1.0 / n_s_a[state, action]
        Q[state, action] += alpha * (result_sum - Q[state, action])

    if i % 500 == 0 and i is not 0:
        print("Success rate: " + str(sum(rList) / i))

print("Success rate: " + str(sum(rList)/num_episodes))

env.close()