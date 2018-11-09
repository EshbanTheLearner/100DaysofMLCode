import numpy as np
import time
import gym

def execute(env, policy, episodeLength=100, render=False):
	totalReward = 0
	start = env.reset()
	for t in range(episodeLength):
		if render:
			env.render()
		action = policy[start]
		start, reward, done, _ = env.step(action)
		totalReward += reward
		if done:
			break
	return totalReward

def evaluatePolicy(env, policy, n_episodes=100):
	totalReward = 0.0
	for _ in range(n_episodes):
		totalReward += execute(env, policy)
	return totalReward/n_episodes

def gen_random_policy():
	return np.random.choice(4, size=((16)))

if __name__ == '__main__':
	env = gym.make('FrozenLake-v0')

	n_policies = 1000
	startTime = time.time()
	policy_set = [gen_random_policy() for _ in range(n_policies)]
	policy_score = [evaluatePolicy(env, p) for p in policy_set]
	endTime = time.time()

	print('Best Score: %0.2f Time Taken: %4.4f seconds' % (np.max(policy_score), endTime-startTime))