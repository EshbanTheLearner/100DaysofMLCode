from collections import deque
import random
import numpy as np
from model import mlp

class DQNAgent(object):

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = mlp(state_size, action_size)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size=32):
		minibatch = random.sample(self.memory, batch_size)
		
		states = np.array([tup[0][0] for tup in minibatch])
		actions = np.array([tup[1] for tup in minibatch])
		rewards = np.array([tup[2] for tup in minibatch])
		next_states = np.array([tup[3][0] for tup in minibatch])
		done = np.array([tup[4] for tup in minibatch])

		target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
		target[done] = rewards[done]

		target_f = self.model.predict(states)

		target_f[range(batch_size), actions] = target
		self.model.fit(states, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)