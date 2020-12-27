import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from collections import deque
import random

class Model:
	def __init__(self, reward, n_channels=1):
		self.initializer = initializers.RandomNormal(stddev=0.04)
		self.n_channels = n_channels

		self.reward = reward
		self.memory = deque(maxlen=2000)
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.005
		self.tau = 0.05
		self.batch_size = 32

		"""
		Prevent catastrophic forgetting by using separate target model

		TODO: could we just init by making a deep copy?

		The target model outputs the best possible Q-value for the given state
		The normal model outputs action for the given state

		The amount of q-values per state happens to be same than amount of actions
		so the models are of the same size and architecture
		"""
		self.model = self.create_model(self.n_channels)
		self.target_model = self.create_model(self.n_channels)


	def create_model(self, n_channels):
		inputs = keras.Input(shape=(240, 320, n_channels))
		x = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(inputs)
		x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #120x160
		x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #60x80
		x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #30x40
		x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #15x20
		x = layers.Conv2D(256, (2, 3), kernel_initializer=self.initializer,
			activation="relu")(x) #14x18
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #7x9
		x = layers.Conv2D(512, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(256, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(128, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Flatten()(x)
		x = layers.Dense(256, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(256, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(128, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		outputs = layers.Dense(15, activation="tanh")(x)

		model = keras.Model(inputs=inputs, outputs=outputs)
		model.summary()

		model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam(lr=self.learning_rate)
		)

		return model

	def train_batch(self, x, y):
		self.model.train_on_batch(x, y)

	"""
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self, x):

		foo = np.argmax(self.model.predict(x)[0])
		print("foo")
		print(foo)


		prediction = self.model.predict(x)[0]
		action = np.where(prediction > 0.0, True, False).tolist()
		action[14] = prediction[14]*100.0
		return action

	"""
	return: list length of 15: 14 booleans and 1 float
	"""
	def get_random_action(self):
		random_action = random.choices([True, False], k=14)
		random_action.append(random.uniform(-100.0, 100.0))
		return random_action

	"""
	Perform one step;
	- create action
	- get reward
	- update game state (happens in make_action under the hood)

	return: reward (1D float)
	"""
	def step(self, game):
		state = game.get_state()

		#TODO: stack frames
		screen_buf = state.screen_buffer

		"""
		Epsilon-greedy algorithm
		With probability epsilon choose a random action ("explore")
		With probability 1-epsilon choose best known action ("exploit")
		"""

		action = self.predict_action(np.expand_dims(screen_buf,0))

		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			action = self.get_random_action()

		# Intentionally ignore the reward the game gives
		game.make_action(action)

		# Instead, use our own reward system
		reward = self.reward.get_reward(game)

		done = game.is_episode_finished()
		bufu = None
		if not done:
			bufu = game.get_state().screen_buffer
		self.remember(screen_buf, action, reward, bufu, done)

		self.replay()

		return reward


	"""
	Add entry to memory
	"""	
	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	"""
	IF we don't have batch_size amount of samples in memory,
	continue gathering memories and do not train the model yet

	ELSE sample randomly from all memories
	Use experience replay to update the actual model
	with out best known actions for each state
	"""
	def replay(self):
		if len(self.memory) < self.batch_size: 
			return

		samples = random.sample(self.memory, self.batch_size)
		for sample in samples:
			state, action, reward, new_state, done = sample
			
			# Get the Q-value from target model
			target = self.target_model.predict(state)
			print("-aaa")
			print(len(target))
			print(target)
			print("---")
			# This comes from the experience replay algorithm
			# 
			if done:
				target[0][action] = reward
			else:
				# Maximum q-value for s'
				Q_future = max(self.model.predict(new_state)[0])
				target[0][action] = reward + Q_future * self.gamma
			

			# target actions for the state
			# as in, the best known actions for the state ...
			self.model.fit(state, target_actions, epochs=1, verbose=0)

	def save_model(self, filename):
		self.model.save(filename)