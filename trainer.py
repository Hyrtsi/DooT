from reward import Reward
from model import Model
import numpy as np
import tensorflow as tf
import random
from utils import *


class MemoryEntry:
	"""
	Create memory entry
	frame_in:	Input frame
	state_in:	Input state (previous state)
	action_in:	Input action (action taken previously)
	action_out:	Output action (action taken to advance game state)
	reward:		Reward given after game state has been advanced using action_out
	"""
	def __init__(self, frame_in, state_in, action_in, action_out, reward):
		self.frame_in = frame_in
		self.state_in = state_in
		# convert actions into continuous domain
		self.action_in = convert_action_to_continuous(action_in)
		self.action_out = convert_action_to_continuous(action_out)
		self.reward = reward
		self.disc_reward = reward # discounted reward


class MemorySequence:
	def __init__(self, discount_factor=0.95):
		self.sequence = []
		self.discount_factor = discount_factor
		self.reward_cum = 0.0 # cumulative reward, assigned in the end of episode

	def add_entry(self, frame_in, state_in, action_in, action_out, reward):
		self.sequence.append(MemoryEntry(
			frame_in, state_in, action_in, action_out, reward))

	def discount(self):
		# iterate the sequence backwards to pass discounted rewards to previous entries
		for i in range(len(self.sequence)-2, -1, -1):
			self.sequence[i].disc_reward =\
				self.discount_factor*self.sequence[i+1].disc_reward +\
				self.sequence[i].reward

	def get_best_entries(self, n_entries):
		# sort according to discounted reward
		seq_sorted = sorted(self.sequence, key=lambda entry: entry.disc_reward)
		return seq_sorted[-n_entries:]


class Trainer:
	def __init__(self, model, reward):
		self.model = model
		self.reward = reward

		self.memory = [] # list of sequences
		self.replay_episode_interval = 8 # experience replay interval in episodes
		self.replay_n_entries_min = 16 # number of entries used for training from worst sequence
		self.replay_n_entries_delta = 16 # number of entries to increase for better sequences
		
		self.epsilon = 1.0 # probability for random action
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999995

		self.episode_id_prev = -1
		self.episode_reset()
	
	"""
	Reset after an episode
	"""
	def episode_reset(self):
		self.reward.reset()
		self.model.reset_state()
		self.action_prev = get_null_action()

		self.reward_cum = 0.0 # cumulative reward

	"""
	Perform one step;
	- create action
	- get reward
	- update game state (happens in make_action under the hood)

	game:		ViZDoom game object
	episode_id:	Id of the currently running episode
	return: 	reward (1D float)
	"""
	def step(self, game, episode_id):
		if episode_id != self.episode_id_prev:
			self.memory.append(MemorySequence())
			self.episode_id_prev = episode_id

		state_game = game.get_state()

		#TODO: stack frames
		screen_buf = state_game.screen_buffer

		# save the previous model state
		state_prev = self.model.state.copy()

		# advance the model state using new screen buffer and the previously taken action
		self.model.advance(screen_buf, self.action_prev)

		# Epsilon-greedy algorithm
		# With probability epsilon choose a random action ("explore")
		# With probability 1-epsilon choose best known action ("exploit")
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)

		if np.random.random() < self.epsilon:
			# with 90% change just mutate the previous action since usually in Doom there's
			# strong coherency between consecutive actions
			if np.random.random() > 0.1:
				action = mutate_action(self.action_prev, 2)
				action[14] *= 0.95 # some damping to reduce that 360 noscope business
			else:
				action = get_random_action(weapon_switch_prob=(0.45-0.4*self.epsilon))
		else:
			action = self.model.predict_action() # make action predicted from model state

		# Intentionally ignore the reward the game gives
		reward = game.make_action(action)
		#print("game reward: {}".format(reward))

		# Instead, use our own reward system
		reward += self.reward.get_reward(game)
		# update cumulative reward and reward delta
		self.reward_cum += reward;

		# TODO temp
		action_print = np.where(action, 1, 0)
		print("{} {:8.3f} {:8.3f}".format(
			action_print[0:14], action[14], reward), end="\r")
		# TODO end of temp

		# Save the step into active(last in the list) memory sequence
		self.memory[-1].add_entry(screen_buf, state_prev, self.action_prev, action, reward)

		# save the action taken for next step
		self.action_prev = action.copy()

		done = game.is_episode_finished()
		if done:
			print("Episode {} finished, cumulative reward: {:10.5f}, epsilon: {:10.5f}"
				.format(episode_id, self.reward_cum, self.epsilon))
			# save the cumulative reward to the sequence
			self.memory[-1].reward_cum = self.reward_cum

			# reset stuff for the new episode
			self.episode_reset()
			
			self.reward.reset_exploration()

			if (episode_id+1) % self.replay_episode_interval == 0:
				print("================================================================================")
				print("Experience replay interval reached, training...")

				# gather best entries from the memory
				frames_in = []
				states_in = []
				actions_in = []
				actions_out = []

				# use self.replay_n_sequences best sequences
				self.memory.sort(key=lambda sequence: sequence.reward_cum)
				n_entries = self.replay_n_entries_min
				for sequence in self.memory:
					print("{:10.5f} {}".format(sequence.reward_cum, n_entries))
					sequence.discount()
					best = sequence.get_best_entries(n_entries)
					for e in best:
						frames_in.append(e.frame_in)
						states_in.append(e.state_in)
						actions_in.append(e.action_in)
						actions_out.append(e.action_out)
					n_entries += self.replay_n_entries_delta

				# train
				self.model.train(frames_in, states_in, actions_in, actions_out)

				# clear memory
				self.memory = []
