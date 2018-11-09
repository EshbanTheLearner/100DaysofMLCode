import numpy as np 
import gym
import time

def execute(env, policy, gamma=1.0, render=False):
	start = env.reset()
	totalReward = 0
	stepIndex = 0
	while True:
		if render:
			env.render()
		start, reward, done, _ = env.step(int(policy[start]))
		totalReward += (gamma ** stepIndex * reward)
		stepIndex = 1
		if done:
			break
	return totalReward

def evaluatePolicy(env, policy, gamma=1.0, n=100):
	scores = [execute(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)

def extractPolicy(v, gamma=1.0):
	policy = np.zeros(env.env.nS)
	for s in range(env.env.nS):
		q_sa = np.zeros(env.env.nA)
		for a in range(env.env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p,s_,r,_ in env.env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy

def calcPolicyValue(env, policy, gamma=1.0):
	value = np.zeros(env.env.nS)
	eps = 1e-10
	while True:
		previousValue = np.copy(value)
		for states in range(env.env.nS):
			policy_a = policy[states]
			value[states] = sum([p * (r + gamma * previousValue[s_]) for p,s_,r,_ in env.env.P[states][policy_a]])
		if (np.sum((np.fabs(previousValue-value))) <= eps):
			#print('Values Converged')
			break
	return value

def policyIteration(env, gamma=1.0):
	policy = np.random.choice(env.env.nA, size=(env.env.nS))
	maxIterations = 1000
	gamma = 1.0
	for i in range(maxIterations):
		oldPolicyValue = calcPolicyValue(env, policy, gamma)
		newPolicy = extractPolicy(oldPolicyValue, gamma)
		if (np.all(policy == newPolicy)):
			print('Policy Iteration Converged at %d'%(i+1))
			break
		policy = newPolicy
	return policy

if __name__ == '__main__':
	
	env_name = 'FrozenLake-v0'
	env = gym.make(env_name)
	startTime = time.time()
	optimalPolicy = policyIteration(env, gamma=1.0)
	scores = evaluatePolicy(env, optimalPolicy, gamma=1.0)
	endTime = time.time()
	print('Best Score: %0.2f Time Taken: %4.4f seconds' % (np.max(scores), endTime-startTime))
