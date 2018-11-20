import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from utils import checkpoint, mkdir, moving_average

GAME = 'Taxi-v2'
env = gym.make(GAME)
CHECKPOINT_DIR = 'checkpoints'
MAX_STEPS = env.spec.timestep_limit

NUM_EPISODES = 50000
GAMMA = 0.95
START_ALPHA = 0.1
ALPHA_TAPER = 0.01
START_EPSILON = 1.0
EPSILON_TAPER = 0.01

obs_dim = env.observation_space.n
action_dim = env.action_space.n

Q = np.zeros((obs_dim, action_dim))

state_visit_counts = {}
update_counts = np.zeros((obs_dim, action_dim), dtype=np.dtype(int))


def update_Q(prev_state, action, reward, cur_state):

	alpha = START_ALPHA / (1.0+update_counts[prev_state][action]*ALPHA_TAPER)
	update_counts[prev_state][action] += 1
	Q[prev_state][action] += \
		alpha * (reward + GAMMA * max(Q[cur_state]) - Q[prev_state][action])

def epsilon_action(s, eps=START_EPSILON):
	if random.random() < (1 - eps):
		return np.argmax(Q[s])
	else:
		return env.action_space.sample()

if __name__ == '__main__':
	
	print("\nObservation\n--------------------------------")
	print("Shape :", obs_dim)
	print("\nAction\n--------------------------------")
	print("Shape :", action_dim, "\n")

	total_reward = 0
	deltas = []

	for episode in range(NUM_EPISODES + 1):
		eps = START_EPSILON / (1.0 + episode * EPSILON_TAPER)

		if episode%10000 == 0:
			cp_file = checkpoint(Q, CHECKPOINT_DIR, GAME, episode)
			print('Saved Checkpoint to: ', cp_file)

		biggest_change = 0
		curr_state = env.reset()
		for step in range(MAX_STEPS):
			prev_state = curr_state
			state_visit_counts[prev_state] = state_visit_counts.get(prev_state,0)+1
			action = epsilon_action(curr_state, eps)
			curr_state, reward, done, info = env.step(action)
			total_reward += reward
			old_qsa = Q[prev_state][action]
			update_Q(prev_state, action, reward, curr_state)
			biggest_change = max(biggest_change, np.abs(old_qsa - Q[prev_state][action]))
			if done:
				break

		deltas.append(biggest_change)

	mean_state_visits = np.mean(list(state_visit_counts.values()))
	print('Each state was visited on average: ', mean_state_visits, ' times')

	plt.plot(moving_average(deltas, n=1000))
	plt.show()