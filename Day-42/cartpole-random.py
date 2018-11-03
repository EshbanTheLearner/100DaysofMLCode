import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
	observation = env.reset()
	total_reward = 0

	for _ in range(200):
		action = 0 if np.matmul(parameters, observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def train(submit):
	env = gym.make('CartPole-v0')
	if submit:
		env.monitor.start('cartpole-experiments/', force=True)

	counter = 0
	best_params = None
	best_reward = 0
	for _ in range(10000):
		counter += 1
		parameters = np.random.rand(4)*2-1
		reward = run_episode(env, parameters)
		if reward > best_reward:
			best_reward = reward
			best_params = parameters
			if reward == 200:
				break

	if submit:
		for _ in range(100):
			run_episode(env, best_params)
		env.monitor.close()

	return counter

results = []
for _ in range(1000):
	results.append(train(submit=False))

plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

print(np.sum(results)/1000.0)