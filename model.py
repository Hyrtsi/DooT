import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
import random
from utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import gc
import os


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class L2Regularizer(regularizers.Regularizer):
	def __init__(self, strength):
		self.strength = tf.Variable(strength)

	def __call__(self, x):
		return self.strength * tf.reduce_mean(tf.square(x))

class MaxRegularizer(regularizers.Regularizer):
	def __init__(self, strength=1.0, batch_size=8.0):
		self.strength = tf.Variable(strength)
		self.batch_size = tf.Variable(batch_size)

	def __call__(self, x):
		return self.strength * self.batch_size * tf.reduce_max(tf.abs(x))


def loss_image(y_true, y_pred):
	return tf.reduce_mean(tf.abs(y_true - y_pred))


def loss_function_inverse(action_true, action_pred):
	loss = tf.reduce_mean(tf.square(
		tf.math.sign(action_true[:,0:14,:])*0.5 - action_pred[:,0:14,:]))
	loss += tf.reduce_mean(tf.abs(action_true[:,14,:] - action_pred[:,14,:]))
	return loss


class ActionModel:
	def __init__(self, model):
		self.model_state = model.model_state
		self.model_encoding = model.model_encoding
		self.model_reward = model.model_reward

		self.tbptt_length_action = model.tbptt_length_action
		self.action_optimizer = model.action_optimizer

		self.create_action_model(model)

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(model.replay_sample_length, 8, model.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(model.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(model.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, model.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32),
			tf.TensorSpec(shape=(), dtype=tf.float32)
		])
		def train(image_encs, actions, rewards, state_init, i, discount_factor):
			discount_falloff = 1.0 # iterative discount factor
			discount_cum = 0.0
			state = self.model_state([state_init, image_encs[i]], training=False)
			with tf.GradientTape(persistent=True) as gt:
				action = self.model_action(state, training=True)
				reward = self.model_reward([state, action], training=True)
				image_enc = image_encs[i]

				reward_mean = tf.reduce_mean(reward)
				loss_reward = -reward_mean
				loss_reg = 2.0*tf.math.pow(self.model_action.losses[0], 4.0)*tf.abs(reward_mean)

				# simulate forward and predict rewards
				for j in range(1, self.tbptt_length_action):
					image_enc = self.model_encoding([image_enc, state, action], training=False)
					state = self.model_state([state, image_enc], training=False)
					action = self.model_action(state, training=True)
					reward = self.model_reward([state, action], training=True)

					reward_mean = tf.reduce_mean(reward)
					loss_reward -= reward_mean*discount_falloff
					loss_reg += 2.0*tf.math.pow(self.model_action.losses[0], 4.0)*\
						tf.abs(reward_mean)*discount_falloff
					
					discount_cum += discount_falloff
					discount_falloff *= discount_factor # update dc. falloff according to dc. factor
				
				loss_total = (loss_reward + loss_reg) / discount_cum
			
			g_model_action = gt.gradient(loss_total, self.model_action.trainable_variables)
			
			self.action_optimizer.apply_gradients(zip(g_model_action,
				self.model_action.trainable_variables))
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			
			l_norm = 1.0 / self.tbptt_length_action
			return state, g_model_action, loss_total*l_norm, loss_reward*l_norm, loss_reg*l_norm
		
		self.train = train
	

	def create_action_model(self, model):
		self.model_action_i_state = keras.Input(shape=(model.state_size))

		x = model.module_dense(self.model_action_i_state, model.state_size, n2=model.state_size)

		self.model_action_o_action = layers.Dense(15,
			kernel_initializer=model.initializer, 
			activity_regularizer=MaxRegularizer(), use_bias=False, activation="tanh")(x)
		
		self.model_action = keras.Model(
			inputs=self.model_action_i_state,
			outputs=self.model_action_o_action,
			name="model_action")
		#self.model_action.summary()
	

	def save(self, filename):
		self.model_action.save_weights(filename)
	

	def load(self, filename):
		self.model_action.load_weights(filename)
	

	def __call__(self, input, training=True):
		return self.model_action(input, training=training)


class Model:
	def __init__(self, episode_length, n_replay_episodes, n_training_epochs, replay_sample_length):
		self.initializer = initializers.RandomNormal(stddev=0.01)
		self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, clipnorm=0.1, clipvalue=1.0)
		self.action_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.loss_function = keras.losses.MeanSquaredError()
		self.loss_image = loss_image
		#self.loss_action = loss_action

		self.state_size = 256
		self.image_enc_size = 256
		self.tbptt_length_encoder = 8
		self.tbptt_length_backbone = 64
		self.tbptt_length_action = 16

		self.reset_state()
		self.action_predict_step_size = tf.Variable(0.01)
		self.episode_length = episode_length
		self.n_replay_episodes = n_replay_episodes
		self.n_training_epochs = n_training_epochs
		self.replay_sample_length = replay_sample_length

		self.create_image_encoder_model(feature_multiplier=2)
		self.create_image_decoder_model(feature_multiplier=2)

		self.create_state_model()
		self.create_reward_model()
		self.create_encoding_model()
		self.create_inverse_model()
		self.create_inverse_backbone_model()
		self.create_action_models()

		self.define_training_functions()


	def define_training_functions(self):

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 240, 320, 4), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_image_encoder_model(images, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				image_enc = self.model_image_encoder(images[i], training=True)
				state = self.model_state([state_init, image_enc], training=True)
				reward = self.model_reward([state, actions[i]], training=True)
				states = [state]

				loss_reward = self.loss_function(rewards[i], reward)

				for j in range(1, self.tbptt_length_encoder):
					image_enc = self.model_image_encoder(images[i+j], training=True)
					state = self.model_state([state, image_enc], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)
					states.append(state)
					
					# reward loss
					loss_reward += self.loss_function(rewards[i+j], reward)

				actions_pred = self.model_inverse(tf.expand_dims(tf.stack(states, axis=-1), axis=-1),
					training=True)[:,:,:,0]
				actions_true = tf.transpose(actions[i:i+self.tbptt_length_encoder-1], perm=[1, 2, 0])
				
				loss_reward /= self.tbptt_length_encoder
				loss_reward *= 0.1 # TODO TEMP multiplier
				loss_total = loss_reward
				loss_inverse = loss_function_inverse(actions_true, actions_pred)
				loss_inverse *= 10.0 # TODO TEMP multiplier
				loss_total += loss_inverse
			
			g_model_image_encoder = gt.gradient(loss_total, self.model_image_encoder.trainable_variables)
			g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			g_model_inverse = gt.gradient(loss_total, self.model_inverse.trainable_variables)
			g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
			
			state = self.model_state([state_init, self.model_image_encoder(images[i],
				training=False)], training=False)
			
			# l_norm = 1.0 / self.tbptt_length_encoder
			return state, g_model_image_encoder, g_model_state, g_model_inverse, g_model_reward,\
				loss_total, loss_reward, loss_inverse


		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_backbone_inverse(image_encs, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				state= self.model_state([state_init, image_encs[i]], training=True)
				reward = self.model_reward([state, actions[i]], training=True)
				states = [state]

				# loss_total = self.model_state.losses[0]
				loss_reward = self.loss_function(rewards[i], reward)
				loss_forward = tf.zeros_like(loss_reward)

				for j in range(1, self.tbptt_length_backbone):
					image_enc_pred = self.model_encoding([image_encs[i+j-1], state, actions[i+j-1]])

					state = self.model_state([state, image_encs[i+j]], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)
					states.append(state)
					
					# reward loss
					loss_reward += self.loss_function(rewards[i+j], reward)
					# forward loss
					loss_forward += tf.reduce_mean(tf.abs(image_encs[i+j] - image_enc_pred))

				actions_pred = self.model_inverse_backbone(
					tf.expand_dims(tf.stack(states, axis=-1), axis=-1), training=True)[:,:,:,0]
				actions_true = tf.transpose(actions[i:i+self.tbptt_length_backbone-1], perm=[1, 2, 0])
				
				loss_reward /= self.tbptt_length_backbone
				loss_reward *= 0.1 # TODO TEMP multiplier

				loss_forward /= (self.tbptt_length_backbone-1)

				loss_total = loss_reward + loss_forward
				loss_inverse = loss_function_inverse(actions_true, actions_pred)
				loss_inverse *= 10.0 # TODO TEMP multiplier
				loss_total += loss_inverse
			
			g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			g_model_inverse_backbone = gt.gradient(loss_total,
				self.model_inverse_backbone.trainable_variables)
			g_model_forward = gt.gradient(loss_total, self.model_encoding.trainable_variables)
			g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			
			# l_norm = 1.0 / self.tbptt_length_encoder
			return state, g_model_state, g_model_inverse_backbone, g_model_forward, g_model_reward,\
				loss_total, loss_reward, loss_forward, loss_inverse

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_backbone(image_encs, actions, rewards, state_init, i):
			discount_factor = 1.0
			discount_cum = 0.0
			with tf.GradientTape(persistent=True) as gt:
				state = self.model_state([state_init, image_encs[i]], training=True)
				# reward = self.model_reward([state, actions[i]], training=True)
				image_enc = image_encs[i]

				# loss_total = self.model_image_encoder.losses[0] + self.model_state.losses[0]
				# loss_reward = self.loss_function(rewards[i], reward)
				loss_forward = tf.zeros(shape=())

				for j in range(1, self.tbptt_length_backbone):
					image_enc = self.model_encoding([image_enc, state, actions[i+j-1]], training=True)
					state = self.model_state([state, image_enc], training=True)
					# reward = self.model_reward([state, actions[i+j]], training=True)

					loss_enc_iter = tf.reduce_mean(tf.abs(image_encs[i+j] - image_enc))
					loss_forward += loss_enc_iter * discount_factor
					# loss_reward += self.loss_function(rewards[i+j], reward) * discount_factor
					# loss_total += self.model_image_encoder.losses[0] + self.model_state.losses[0]

					# discount falloff according to prediction error
					discount_cum += discount_factor
					discount_factor *= tf.clip_by_value(1.0-loss_enc_iter, 1.0e-10, 1.0)
				
				# loss_forward *= 10.0 # TODO TEMP?
				# loss_reward /= discount_cum # normalize by cumulative discount
				loss_forward /= discount_cum
				# loss_total += loss_forward#+ loss_reward
			
			g_model_encoding = gt.gradient(loss_forward, self.model_encoding.trainable_variables)
			# g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			# g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
		
			# self.optimizer.apply_gradients(zip(g_model_encoding,
			# 	self.model_encoding.trainable_variables))
			# self.optimizer.apply_gradients(zip(g_model_state,
			# 	self.model_state.trainable_variables))
			# self.optimizer.apply_gradients(zip(g_model_reward,
			# 	self.model_reward.trainable_variables))
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			return state, g_model_encoding, loss_forward, discount_cum
		

		self.train_image_encoder_model = train_image_encoder_model
		self.train_backbone_inverse = train_backbone_inverse
		self.train_backbone = train_backbone

	
	def module_dense(self, x, n, x2=None, n2=None, alpha=0.001, act=None):
		use_shortcut = n == x.shape[1]

		if use_shortcut:
			y = x

		# enable concatenation if auxiliary input is provided
		if x2 is not None:
			x = layers.Concatenate()([x, x2])
		
		# double layer model
		if n2 is not None:
			x = layers.Dense(n2, kernel_initializer=self.initializer, use_bias=False)(x)
			x = layers.BatchNormalization(axis=-1,
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
			x = activations.relu(x, alpha=alpha)

		x = layers.Dense(n, kernel_initializer=self.initializer, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
		x = activations.relu(x, alpha=alpha)

		if use_shortcut:
			x = layers.Add()([x, y])
		
		if act is not None:
			x = act(x)
		
		return x
		
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", alpha=0.001):

		#shortcut by avg pooling
		pool_x_size = s1[0]*s2[0]
		pool_y_size = s1[1]*s2[1]
		skip_x_kernel = 1
		skip_y_kernel = 1
		if p1 == "valid":
			skip_x_kernel += k1[0]-1
			skip_y_kernel += k1[1]-1
		if p2 == "valid":
			skip_x_kernel += k2[0]-1
			skip_y_kernel += k2[1]-1
		
		y = x
		if pool_x_size != 1 or pool_y_size != 1:
			y = layers.AveragePooling2D((pool_x_size, pool_y_size))(y)	
		if n2 != x.shape[3] or skip_x_kernel > 1 or skip_y_kernel > 1:
			y = layers.Conv2D(n2, (skip_x_kernel, skip_y_kernel),
				kernel_initializer=self.initializer, use_bias=False, padding="valid")(y)
			y = layers.BatchNormalization(axis=-1,
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(y)

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
		x = activations.relu(x, alpha=alpha)

		return layers.Add()([x, y])
	

	def module_deconv(self, x, n1, n2, k1=(4,4), k2=(2,2), s1=(2,2), s2=(1,1),
		p1="same", p2="same", bn2=True, alpha=0.001, act=None):

		#shortcut by linear upsampling
		pool_x_size = s1[0]*s2[0]
		pool_y_size = s1[1]*s2[1]
		if p1 == "valid":
			pool_x_size += k1[0]-1
			pool_y_size += k1[1]-1
		if p2 == "valid":
			pool_x_size += k2[0]-1
			pool_y_size += k2[1]-1
		y = layers.UpSampling2D((pool_x_size, pool_y_size), interpolation="bilinear")(x)
		y = layers.Conv2D(n2, (1, 1), kernel_initializer=self.initializer, use_bias=False)(y)
		y = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(y)

		x = layers.Conv2DTranspose(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2DTranspose(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Add()([x, y])

		if act is not None:
			x = act(x)
		
		return x
	

	def module_inception(self, x, n=[], k=[], alpha=0.001):
		n_total = sum(n)

		y = x
		if n_total != x.shape[3]:
			y = layers.Conv2D(n_total, (1, 1),
				kernel_initializer=self.initializer, use_bias=False)(y)
			y = layers.BatchNormalization(axis=-1,
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(y)

		xv = []
		for i in range(len(n)):
			xv.append(layers.Conv2D(n[i], k[i], padding="same",
				kernel_initializer=self.initializer, use_bias=False)(x))
			xv[-1] = layers.BatchNormalization(axis=-1,
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.01))(xv[-1])
			xv[-1] = activations.relu(xv[-1], alpha=alpha)

		x = layers.concatenate(xv)

		return layers.Add()([x, y])
	

	def create_image_encoder_model(self, feature_multiplier=1):
		self.model_image_encoder_i_image = keras.Input(shape=(240, 320, 4))

		# camera branch
		x = self.module_conv(self.model_image_encoder_i_image[:,:,:,0:3],
			4*feature_multiplier, 8*feature_multiplier,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2)) #80x80
		x = self.module_conv(x, 16*feature_multiplier, 16*feature_multiplier) #40x40
		x = self.module_conv(x, 32*feature_multiplier, 32*feature_multiplier) #20x20
		x = self.module_conv(x, 64*feature_multiplier, 64*feature_multiplier) #10x10
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier, k2=(1,1)) #5x5
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier,
			s1=(1,1), p1="valid", p2="valid") #1x1
		x = layers.Flatten()(x)

		# automap branch
		y = self.module_conv(self.model_image_encoder_i_image[:,:,:,3:4],
			2*feature_multiplier, 2*feature_multiplier,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2)) #80x80
		y = self.module_conv(y, 4*feature_multiplier, 4*feature_multiplier) #40x40
		y = self.module_conv(y, 8*feature_multiplier, 8*feature_multiplier) #20x20
		y = self.module_conv(y, 16*feature_multiplier, 16*feature_multiplier) #10x10
		y = self.module_conv(y, 32*feature_multiplier, 32*feature_multiplier, k2=(1,1)) #5x5
		y = self.module_conv(y, 64*feature_multiplier, 64*feature_multiplier,
			s1=(1,1), p1="valid", p2="valid") #1x1
		y = layers.Flatten()(y)

		self.model_image_encoder_o_image_enc = self.module_dense(
			x, self.image_enc_size,
			x2=y, n2=self.image_enc_size,
			act=layers.Activation(activations.tanh, activity_regularizer=L2Regularizer(1.0e-2)))

		self.model_image_encoder = keras.Model(
			inputs=self.model_image_encoder_i_image,
			outputs=self.model_image_encoder_o_image_enc,
			name="model_image_encoder")
		# self.model_image_encoder.summary()
	

	def create_image_decoder_model(self, feature_multiplier=1):
		self.model_image_decoder_i_image_enc = keras.Input(shape=(self.image_enc_size))
		x = layers.Reshape((1, 1, -1))(self.model_image_decoder_i_image_enc)

		x = self.module_deconv(x, 128*feature_multiplier, 128*feature_multiplier,
			k1=(3,3), s1=(1,1), k2=(3,3), s2=(1,1), p1="valid", p2="valid", alpha=1.0e-6) #5x5
		x = self.module_deconv(x, 128*feature_multiplier, 64*feature_multiplier,
			k1=(3,4), s1=(3,4), k2=(3,3), s2=(1,1), alpha=1.0e-6) #20x15
		x = self.module_deconv(x, 64*feature_multiplier, 32*feature_multiplier, alpha=1.0e-6) #40x30
		x = self.module_deconv(x, 32*feature_multiplier, 16*feature_multiplier, alpha=1.0e-6) #80x60
		x = self.module_deconv(x, 16*feature_multiplier, 8*feature_multiplier, k2=(3,3), alpha=1.0e-6) #160x120

		self.model_image_decoder_o_image = self.module_deconv(x, 8*feature_multiplier, 4,
			act=layers.Activation(activations.sigmoid), k2=(3,3), alpha=1.0e-6)

		self.model_image_decoder = keras.Model(
			inputs=self.model_image_decoder_i_image_enc,
			outputs=self.model_image_decoder_o_image,
			name="model_image_decoder")
		#self.model_image_decoder.summary()


	def create_state_model(self):
		self.model_state_i_state = keras.Input(shape=(self.state_size))
		self.model_state_i_image_enc = keras.Input(shape=(self.image_enc_size))
		
		# concatenate image encoding and layer 
		#x = layers.concatenate([self.model_state_i_state, self.model_state_i_image_enc])

		# first layer uses concatenative dense module
		x = self.module_dense(self.model_state_i_state, self.state_size,
			x2=self.model_state_i_image_enc, n2=self.state_size+self.image_enc_size)
		#x = self.module_dense(x, self.state_size)

		# state output
		s = self.module_dense(x, self.state_size)
		s = layers.Dense(self.state_size, kernel_initializer=self.initializer,
			use_bias=False, activation="tanh", activity_regularizer=L2Regularizer(5.0e-2))(s)
		
		# gate output value for previous state feedthrough
		#g1 = self.module_dense(x, self.state_size)
		g1 = layers.Dense(self.state_size, kernel_initializer=self.initializer,
			use_bias=False, activation="sigmoid")(x)#(g1)

		# gate output value for new state (1 - g1)
		g2 = layers.Lambda(lambda x: 1.0 - x)(g1)

		# gated new state
		self.model_state_o_state = layers.Add()([\
			layers.Multiply()([self.model_state_i_state, g1]),
			layers.Multiply()([s, g2])
		]) # new state mixed from old and new according to the gate values

		self.model_state = keras.Model(
			inputs=[self.model_state_i_state, self.model_state_i_image_enc],
			outputs=self.model_state_o_state,
			name="model_state")
		#self.model_state.summary()


	def create_reward_model(self):
		self.model_reward_i_state = keras.Input(shape=(self.state_size))
		self.model_reward_i_action = keras.Input(shape=(15))

		x = self.module_dense(self.model_reward_i_state, self.state_size,
			x2=self.model_reward_i_action, n2=self.state_size)
		x = self.module_dense(x, self.state_size, n2=self.state_size)
		x = self.module_dense(x, self.state_size, n2=self.state_size)

		self.model_reward_o_reward_step = layers.Dense(1,
			kernel_initializer=self.initializer, use_bias=False)(x)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=[self.model_reward_o_reward_step],
			name="model_reward")
		# self.model_reward.summary()
	

	# predict encoding of next image from state and action
	def create_encoding_model(self):
		self.model_encoding_i_image_enc = keras.Input(shape=(self.image_enc_size))
		self.model_encoding_i_state = keras.Input(shape=(self.state_size))
		self.model_encoding_i_action = keras.Input(shape=(15))

		y = layers.concatenate([self.model_encoding_i_state, self.model_encoding_i_action])

		x = self.module_dense(
			self.model_encoding_i_image_enc, self.image_enc_size,
			x2=y, n2=self.state_size+self.image_enc_size)
		x = self.module_dense(x, self.image_enc_size, n2=self.image_enc_size)

		self.model_encoding_o_image_enc = layers.Dense(self.image_enc_size,
			kernel_initializer=self.initializer, activation="tanh")(x)

		self.model_encoding = keras.Model(
			inputs=[self.model_encoding_i_image_enc,
				self.model_encoding_i_state,
				self.model_encoding_i_action],
			outputs=[self.model_encoding_o_image_enc],
			name="model_encoding")
		# self.model_encoding.summary()


	def create_inverse_model(self):
		l_e = self.tbptt_length_encoder # alias for easier usage
		s8 = int(self.state_size/8)

		self.model_inverse_i_states = keras.Input(shape=(self.state_size, l_e, 1))

		i_n = [8, 8, 8, 8]
		n = sum(i_n)
		x = self.module_inception(self.model_inverse_i_states, i_n, [(1,3), (1,5), (1,7), (1,9)])
		x = self.module_inception(x, i_n, [(1,1), (1,3), (1,5), (1,7)])

		x = self.module_conv(x, n, n, k1=(1,3), k2=(1,2),
			s1=(1,1), s2=(1,1), p1="same", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		self.model_inverse_o_actions = layers.Conv2D(1,
			kernel_size=(int(self.state_size/4)-14,1), padding="valid", activation="tanh")(x)

		self.model_inverse = keras.Model(
			inputs=self.model_inverse_i_states,
			outputs=self.model_inverse_o_actions,
			name="model_inverse")
		# self.model_inverse.summary()
	

	def create_inverse_backbone_model(self):
		l_e = self.tbptt_length_backbone # alias for easier usage
		s8 = int(self.state_size/8)

		self.model_inverse_backbone_i_states = keras.Input(shape=(self.state_size, l_e, 1))

		i_n = [8, 8, 8, 8]
		n = sum(i_n)
		x = self.module_inception(self.model_inverse_backbone_i_states,
			i_n, [(1,3), (1,5), (1,9), (1,17)])
		x = self.module_inception(x, i_n, [(1,3), (1,5), (1,9), (1,17)])
		x = self.module_inception(x, i_n, [(1,3), (1,5), (1,9), (1,17)])

		x = self.module_conv(x, n, n, k1=(1,3), k2=(1,2),
			s1=(1,1), s2=(1,1), p1="same", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])

		x = self.module_conv(x, n, n, k1=(s8+1,1), k2=(s8+1,1),
			s1=(1,1), s2=(1,1), p1="valid", p2="valid")
		x = self.module_inception(x, i_n, [(1,1), (3,3), (5,5), (7,7)])
		
		self.model_inverse_backbone_o_actions = layers.Conv2D(1,
			kernel_size=(int(self.state_size/4)-14,1), padding="valid", activation="tanh")(x)

		self.model_inverse_backbone = keras.Model(
			inputs=self.model_inverse_backbone_i_states,
			outputs=self.model_inverse_backbone_o_actions,
			name="model_inverse_backbone")
		# self.model_inverse_backbone.summary()


	def create_action_models(self):
		self.models_action = []
		for i in range(self.n_replay_episodes):
			self.models_action.append(ActionModel(self))


	def advance(self, image, action_prev):
		# preprocess image and previous action
		image = tf.convert_to_tensor(image, dtype=tf.float32) * 0.0039215686274509803 # 1/255
		action_prev = tf.expand_dims(action_prev,0)

		# predict encoding from previous image encoding and state
		image_enc_pred = self.model_encoding([
			self.image_enc, self.state, action_prev], training=False)

		# update image encoding and state
		self.image_enc = self.model_image_encoder(tf.expand_dims(image, 0), training=False)
		self.state = self.model_state([self.state, self.image_enc], training=False)

		# return curiosity reward - difference between predicted and real image encoding
		return tf.reduce_mean(tf.abs(self.image_enc[0] - image_enc_pred[0]))

	"""
	Reset state (after an episode)
	"""
	def reset_state(self):
		self.image_enc = tf.zeros((1, self.image_enc_size))
		self.state = tf.zeros((1, self.state_size))

	"""
	Predict action from the state of the model
	"""
	def predict_action(self, model_id, epsilon=0.0):
		state_input = (1.0-epsilon)*self.state +\
			epsilon*tf.random.uniform((1, self.state_size), -1.0, 1.0)
		action = self.models_action[model_id](state_input, training=False)[0]

		return action.numpy()

	def predict_worst_action(self):
		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward = self.model_reward([self.state, action], training=False)
			action_grad = g.gradient(reward, action)
			action_grad /= tf.math.reduce_std(action_grad) + 1e-8
			action = action - action_grad*self.action_predict_step_size
		
		action_max = tf.math.reduce_max(tf.abs(action))
		self.action_predict_step_size = ((1.0 / action_max)*0.1 + 0.9)*self.action_predict_step_size

		return convert_action_to_mixed(action[0])


	def train(self, memory):
		n_sequences = memory.images.shape[1]

		image_encs = tf.Variable(tf.zeros((self.replay_sample_length, n_sequences,
			self.image_enc_size)))

		for e in range(self.n_training_epochs):
			# compute initial states
			#memory.compute_states(self.model_state, self.model_image_encoder)
			
			images, actions, rewards, state_init = memory.get_sample(self.replay_sample_length,
				self.model_state, self.model_image_encoder)

			# train the image encodet model (and reward model, 1st phase)
			state_prev = state_init
			loss_total = 0.0
			loss_reward = 0.0
			loss_inverse = 0.0
			g_model_image_encoder = None
			g_model_state = None
			g_model_inverse = None
			g_model_reward = None
			for i in range(self.replay_sample_length-self.tbptt_length_encoder):
				state_prev, gi_model_image_encoder, gi_model_state, gi_model_inverse, gi_model_reward,\
					loss_total_tf, loss_reward_tf, loss_inverse_tf =\
					self.train_image_encoder_model(images, actions, rewards,
					state_prev, tf.convert_to_tensor(i))
				
				loss_total += loss_total_tf.numpy()
				loss_reward += loss_reward_tf.numpy()
				loss_inverse += loss_inverse_tf.numpy()

				print("Epoch {:3d} - Training image encoder model ({}/{}) l_t: {:8.5f} l_r: {:8.5f} l_i: {:8.5f}".format(
					e, i+self.tbptt_length_encoder+1, self.replay_sample_length,
					loss_total/(i+1), loss_reward/(i+1), loss_inverse/(i+1)),
					end="\r")
				
				if g_model_image_encoder is None:
					g_model_image_encoder = gi_model_image_encoder
				else:
					g_model_image_encoder = [a+b for a,b in zip(g_model_image_encoder, gi_model_image_encoder)]

				if g_model_state is None:
					g_model_state = gi_model_state
				else:
					g_model_state = [a+b for a,b in zip(g_model_state, gi_model_state)]

				if g_model_inverse is None:
					g_model_inverse = gi_model_inverse
				else:
					g_model_inverse = [a+b for a,b in zip(g_model_inverse, gi_model_inverse)]

				if g_model_reward is None:
					g_model_reward = gi_model_reward
				else:
					g_model_reward = [a+b for a,b in zip(g_model_reward, gi_model_reward)]
			print("")
			
			self.optimizer.apply_gradients(zip(g_model_image_encoder,
				self.model_image_encoder.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_inverse,
				self.model_inverse.trainable_variables))


			for i in range(self.replay_sample_length):
				image_encs[i].assign(self.model_image_encoder(images[i], training=False))
				print("Computing image encodings... {} / {}      ".format(i+1,
				self.replay_sample_length), end="\r")
			
			# train backbone with inverse model
			state_prev = state_init
			loss_total = 0.0
			loss_reward = 0.0
			loss_forward = 0.0
			loss_inverse = 0.0
			g_model_inverse_backbone = None
			g_model_forward = None
			for i in range(self.replay_sample_length-self.tbptt_length_backbone):
				state_prev,\
					gi_model_state, gi_model_inverse_backbone, gi_model_forward, gi_model_reward,\
					loss_total_tf, loss_reward_tf, loss_forward_tf, loss_inverse_tf =\
					self.train_backbone_inverse(image_encs, actions, rewards,
					state_prev, tf.convert_to_tensor(i))
				loss_total += loss_total_tf.numpy()
				loss_reward += loss_reward_tf.numpy()
				loss_forward += loss_forward_tf.numpy()
				loss_inverse += loss_inverse_tf.numpy()
				print("Epoch {:3d} - Training the backbone ({}/{}) l_t: {:8.5f} l_r: {:8.5f} l_f: {:8.5f} l_i: {:8.5f}".format(
					e, i+self.tbptt_length_backbone+1, self.replay_sample_length,
					loss_total/(i+1), loss_reward/(i+1), loss_forward/(i+1), loss_inverse/(i+1)),
					end="\r")
				
				if g_model_state is None:
					g_model_state = gi_model_state
				else:
					g_model_state = [a+b for a,b in zip(g_model_state, gi_model_state)]

				if g_model_inverse_backbone is None:
					g_model_inverse_backbone = gi_model_inverse_backbone
				else:
					g_model_inverse_backbone = [a+b for a,b in zip(g_model_inverse_backbone, gi_model_inverse_backbone)]

				if g_model_forward is None:
					g_model_forward = gi_model_forward
				else:
					g_model_forward = [a+b for a,b in zip(g_model_forward, gi_model_forward)]

				if g_model_reward is None:
					g_model_reward = gi_model_reward
				else:
					g_model_reward = [a+b for a,b in zip(g_model_reward, gi_model_reward)]

			print("")

			# train the forward model
			state_prev = state_init
			# loss_total = 0.0
			# loss_reward = 0.0
			loss_forward = 0.0
			discount_cum = 0.0 # discount_cum signifies successful prediction falloff volume - "confidence"
			for i in range(self.replay_sample_length-self.tbptt_length_backbone):
				state_prev, gi_model_encoding, loss_forward_tf, discount_cum_tf =\
					self.train_backbone(image_encs, actions, rewards, state_prev,
					tf.convert_to_tensor(i))
				# loss_total += loss_total_tf.numpy()
				# loss_reward += loss_reward_tf.numpy()
				loss_forward += loss_forward_tf.numpy()
				discount_cum += discount_cum_tf.numpy()
				print("Epoch {:3d} - Training forward model ({}/{}) l_f: {:8.5f} d_c: {:8.5f}".format(
					e, i+self.tbptt_length_backbone+1, self.replay_sample_length,
					loss_forward/(i+1), discount_cum/(i+1)), end="\r")

				if g_model_forward is None:
					g_model_forward = gi_model_forward
				else:
					g_model_forward = [a+b for a,b in zip(g_model_forward, gi_model_forward)]
			print("")
			
			# train the action (policy) models
			train_discount_factor = np.math.exp(-1.0/(discount_cum/(i+1))) # use prediction confidence as a basis for dc. factor
			for j in range(self.n_replay_episodes):
				state_prev = state_init
				loss_total = 0.0
				loss_reward = 0.0
				loss_reg = 0.0
				g_model_action = None
				for i in range(self.replay_sample_length):
					state_prev, gi_model_action, loss_total_tf, loss_reward_tf, loss_reg_tf =\
						self.models_action[j].train(image_encs, actions, rewards, state_prev,
						tf.convert_to_tensor(i), tf.convert_to_tensor(train_discount_factor))
					loss_total += loss_total_tf.numpy()
					loss_reward += loss_reward_tf.numpy()
					loss_reg += loss_reg_tf.numpy()
					print("Epoch {:3d} - Training action model {} ({}/{}) l_t: {:8.5f} l_rw: {:8.5f} l_rg: {:8.5f}".format(
						e, j, i+1, self.replay_sample_length,
						loss_total/(i+1), loss_reward/(i+1), loss_reg/(i+1)), end="\r")
					
					if g_model_action is None:
						g_model_action = gi_model_action
					else:
						g_model_action = [a+b for a,b in zip(g_model_action, gi_model_action)]

				self.optimizer.apply_gradients(zip(g_model_action,
					self.models_action[j].model_action.trainable_variables))
			print("")
			
			self.optimizer.apply_gradients(zip(g_model_state,
				self.model_state.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_inverse_backbone,
				self.model_inverse_backbone.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_reward,
				self.model_reward.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_forward,
				self.model_encoding.trainable_variables))
			
			self.save_model("model", "model")

			del images, actions, rewards, state_init
			gc.collect()
	
	#@tf.function
	def train_image_autoencoder(self, image):
		n_epochs = 8

		n_entries = image.shape[0]

		image = tf.convert_to_tensor(image)
		for e in range(n_epochs):
			loss_sum = 0.0
			for i in range(int(n_entries)):
				with tf.GradientTape(persistent=True) as gt:
					image_enc = self.model_image_encoder(image[i])
					image_pred = self.model_image_decoder(image_enc)
					loss = self.loss_image(image[i], image_pred)
					loss_sum += loss.numpy()
				
				g_model_image_encoder = gt.gradient(loss, self.model_image_encoder.trainable_variables)
				g_model_image_decoder = gt.gradient(loss, self.model_image_decoder.trainable_variables)
				
				print("{:2d} {:4d}/{:4d} ({})".format(e, i, n_entries, loss_sum/(i+1)), end="\r")
			
				self.optimizer.apply_gradients(zip(g_model_image_encoder,
					self.model_image_encoder.trainable_variables))
				self.optimizer.apply_gradients(zip(g_model_image_decoder,
					self.model_image_decoder.trainable_variables))
			
				if i % 4 == 0:
					img = cv2.hconcat([image[i,e].numpy(), image_pred[e].numpy()])
					cv2.imshow("target / prediction", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
					cv2.waitKey(10)


	def save_model(self, folder_name, model_name):
		# backup
		ret = os.system("cp {}/* {}_backup/".format(folder_name, folder_name))

		print("Saving model with prefix: {}/{}".format(folder_name, model_name))
		self.model_image_encoder.save_weights("{}/{}_image_encoder.h5".format(folder_name, model_name))
		self.model_image_decoder.save_weights("{}/{}_image_decoder.h5".format(folder_name, model_name))
		self.model_state.save_weights("{}/{}_state.h5".format(folder_name, model_name))
		for i in range(self.n_replay_episodes):
			self.models_action[i].save("{}/{}_action_{}.h5".format(folder_name, model_name, i))
		self.model_reward.save_weights("{}/{}_reward.h5".format(folder_name, model_name))
		self.model_encoding.save_weights("{}/{}_encoding.h5".format(folder_name, model_name))
		self.model_inverse.save_weights("{}/{}_inverse.h5".format(folder_name, model_name))
		self.model_inverse_backbone.save_weights("{}/{}_inverse_backbone.h5".format(folder_name, model_name))
	
	def load_with_backup(self, model, filename, backup_filename):
		try:
			model.load_weights(filename)
		except:
			model.load_weights(backup_filename)
		
	def load_model(self, folder_name, model_name):
		print("Loading model: {}/{}".format(folder_name, model_name))
		self.load_with_backup(self.model_image_encoder,
			"{}/{}_image_encoder.h5".format(folder_name, model_name),
			"{}_backup/{}_image_encoder.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_image_decoder,
			"{}/{}_image_decoder.h5".format(folder_name, model_name),
			"{}_backup/{}_image_decoder.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_state,
			"{}/{}_state.h5".format(folder_name, model_name),
			"{}_backup/{}_state.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_reward,
			"{}/{}_reward.h5".format(folder_name, model_name),
			"{}_backup/{}_reward.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_encoding,
			"{}/{}_encoding.h5".format(folder_name, model_name),
			"{}_backup/{}_encoding.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_inverse,
			"{}/{}_inverse.h5".format(folder_name, model_name),
			"{}_backup/{}_inverse.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_inverse_backbone,
			"{}/{}_inverse_backbone.h5".format(folder_name, model_name),
			"{}_backup/{}_inverse_backbone.h5".format(folder_name, model_name))

		for i in range(self.n_replay_episodes):
			self.load_with_backup(self.models_action[i].model_action,
				"{}/{}_action_{}.h5".format(folder_name, model_name, i),
				"{}_backup/{}_action_{}.h5".format(folder_name, model_name, i))
	
	def create_copy(self):
		model_copy = Model(
			self.episode_length,
			self.n_replay_episodes,
			self.n_training_epochs,
			self.replay_sample_length)
		
		model_copy.model_image_encoder.set_weights(self.model_image_encoder.get_weights())
		model_copy.model_image_decoder.set_weights(self.model_image_decoder.get_weights())
		model_copy.model_state.set_weights(self.model_state.get_weights())
		for i in range(self.n_replay_episodes):
			model_copy.models_action[i].model_action.set_weights(self.models_action[i].model_action.get_weights())
		model_copy.model_reward.set_weights(self.model_reward.get_weights())
		model_copy.model_encoding.set_weights(self.model_encoding.get_weights())
		model_copy.model_inverse.set_weights(self.model_inverse.get_weights())
		model_copy.model_inverse_backbone.set_weights(self.model_inverse_backbone.get_weights())

		return model_copy