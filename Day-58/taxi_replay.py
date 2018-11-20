import sys
import numpy as np
import gym

GAME = 'Taxi-v2'
env = gym.make(GAME)
MAX_STEPS = env.spec.timestep_limit

if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit('Must specify a checkpoint file in CMD')
	cp_file = sys.argv[1]

	Q = np.load(cp_file)
	total_reward = 0
	state = env.reset()
	env.render()

	for step in range(MAX_STEPS):
		prevState = state
		action = np.argmax(Q[state])
		state, reward, done, info = env.step(action)
		total_reward += reward
		env.render()
		if done:
			break


	print('Total Reward: ', total_reward)