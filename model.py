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


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class L2Regularizer(regularizers.Regularizer):
	def __init__(self, strength):
		self.strength = tf.Variable(strength)

	def __call__(self, x):
		return self.strength * tf.reduce_mean(tf.square(x))


def loss_image(y_true, y_pred):
	return tf.reduce_mean(tf.abs(y_true - y_pred))


def loss_action(y_true, y_pred, reward):
	# negative reward scaling for gradient ascent towards better actions
	return tf.reduce_mean(tf.square(y_true-y_pred) * tf.expand_dims(-reward, axis=-1))


class Model:
	def __init__(self, episode_length, n_replay_episodes, n_training_epochs, replay_sample_length):
		self.initializer = initializers.RandomNormal(stddev=0.02)
		self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.action_optimizer = keras.optimizers.SGD(learning_rate=0.0001)
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
		self.create_action_model()
		self.create_reward_model()
		self.create_encoding_model()

		self.define_training_functions()


	def define_training_functions(self):

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 240, 320, 3), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_image_encoder_model(images, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				image_enc = self.model_image_encoder(images[i], training=True)
				state= self.model_state([state_init, image_enc], training=True)
				reward = self.model_reward([state, actions[i]], training=True)

				loss = self.loss_function(rewards[i], reward)
				loss = self.model_image_encoder.losses[0] + self.model_state.losses[0]

				for j in range(1, self.tbptt_length_encoder):
					image_enc = self.model_image_encoder(images[i+j], training=True)
					state= self.model_state([state, image_enc], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)

					# reward loss
					loss += self.loss_function(rewards[i+j], reward)
					# regularization loss
					loss += self.model_image_encoder.losses[0] + self.model_state.losses[0]

				# loss_sum += loss.numpy()
			
			g_model_image_encoder = gt.gradient(loss, self.model_image_encoder.trainable_variables)
			g_model_reward = gt.gradient(loss, self.model_reward.trainable_variables)
		
			self.optimizer.apply_gradients(zip(g_model_image_encoder,
				self.model_image_encoder.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_reward,
				self.model_reward.trainable_variables))
			
			return self.model_state([state_init, self.model_image_encoder(images[i],
				training=False)], training=False)


		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_backbone(image_encs, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				state = self.model_state([state_init, image_encs[i]], training=True)
				reward = self.model_reward([state, actions[i]], training=True)

				loss = self.loss_function(rewards[i], reward)
				loss += self.model_image_encoder.losses[0] + self.model_state.losses[0]
				enc_loss = tf.zeros_like(loss)

				for j in range(1, self.tbptt_length_backbone):
					image_enc_pred = self.model_encoding([state, actions[i+j-1]], training=True)
					state = self.model_state([state, image_enc_pred], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)

					loss += self.loss_function(rewards[i+j], reward)
					loss += self.model_image_encoder.losses[0] + self.model_state.losses[0]
					# image encoding loss
					enc_loss += self.loss_function(image_encs[i+j], image_enc_pred)
				
				loss += enc_loss
				# loss_sum += loss.numpy()
				# enc_loss_sum += enc_loss.numpy()
			
			g_model_encoding = gt.gradient(loss, self.model_encoding.trainable_variables)
			g_model_state = gt.gradient(loss, self.model_state.trainable_variables)
			g_model_reward = gt.gradient(loss, self.model_reward.trainable_variables)
		
			self.optimizer.apply_gradients(zip(g_model_encoding,
				self.model_encoding.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_state,
				self.model_state.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_reward,
				self.model_reward.trainable_variables))
			
			return self.model_state([state_init, image_encs[i]], training=False)
		

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_action_model(image_encs, actions, rewards, state, i):
			state = self.model_state([state, image_encs[i]], training=False)

			with tf.GradientTape(persistent=True) as gt:
				action = self.models_action[self.active_action_model](state, training=True)
				reward_pred = self.model_reward([state, action], training=True)
				loss = -tf.reduce_mean(reward_pred)

				# simulate forward and predict rewards
				for j in range(1, self.tbptt_length_action):
					image_enc_pred = self.model_encoding([state, action], training=False)
					state = self.model_state([state, image_enc_pred], training=False)
					action = self.models_action[self.active_action_model](state, training=True)
					reward_pred = self.model_reward([state, action], training=True)

					loss -= tf.reduce_mean(reward_pred)
			
			g_model_action = gt.gradient(loss, self.models_action[self.active_action_model].trainable_variables)
			
			self.action_optimizer.apply_gradients(zip(g_model_action,
				self.models_action[self.active_action_model].trainable_variables))
			
			return state
		

		self.train_image_encoder_model = train_image_encoder_model
		self.train_backbone = train_backbone
		self.train_action_model = train_action_model

	
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
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
			x = activations.relu(x, alpha=alpha)

		x = layers.Dense(n, kernel_initializer=self.initializer, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
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
		if p1 == "valid":
			pool_x_size += k1[0]-1
			pool_y_size += k1[1]-1
		if p2 == "valid":
			pool_x_size += k2[0]-1
			pool_y_size += k2[1]-1
		y = layers.AveragePooling2D((pool_x_size, pool_y_size))(x)
		y = layers.Conv2D(n2, (1, 1), kernel_initializer=self.initializer, use_bias=False)(y)
		y = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(y)

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
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
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(y)

		x = layers.Conv2DTranspose(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2DTranspose(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Add()([x, y])

		if act is not None:
			x = act(x)
		
		return x
	

	def create_image_encoder_model(self, feature_multiplier=1):
		self.model_image_encoder_i_image = keras.Input(shape=(240, 320, 3))

		x = self.module_conv(self.model_image_encoder_i_image,
			4*feature_multiplier, 8*feature_multiplier,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2)) #80x80
		x = self.module_conv(x, 16*feature_multiplier, 16*feature_multiplier) #40x40
		x = self.module_conv(x, 32*feature_multiplier, 32*feature_multiplier) #20x20
		x = self.module_conv(x, 64*feature_multiplier, 64*feature_multiplier) #10x10
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier, k2=(1,1)) #5x5
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier,
			s1=(1,1), p1="valid", p2="valid") #1x1
		x = layers.Flatten()(x)
		self.model_image_encoder_o_image_enc = self.module_dense(x, self.image_enc_size,
			act=layers.Activation(activations.tanh, activity_regularizer=L2Regularizer(1.0e-2)))

		self.model_image_encoder = keras.Model(
			inputs=self.model_image_encoder_i_image,
			outputs=self.model_image_encoder_o_image_enc,
			name="model_image_encoder")
		#self.model_image_encoder.summary()
	

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

		self.model_image_decoder_o_image = self.module_deconv(x, 8*feature_multiplier, 3,
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
		s = layers.Dense(self.image_enc_size, kernel_initializer=self.initializer,
			use_bias=False, activation="tanh", activity_regularizer=L2Regularizer(2.0e-2))(s)
		
		# gate output value for previous state feedthrough
		#g1 = self.module_dense(x, self.state_size)
		g1 = layers.Dense(self.image_enc_size, kernel_initializer=self.initializer,
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
	

	def create_action_model(self):
		self.models_action = []
		for i in range(self.n_replay_episodes):
			self.model_action_i_state = keras.Input(shape=(self.state_size))

			x = self.module_dense(self.model_action_i_state, self.state_size, n2=self.state_size)

			self.model_action_o_action = layers.Dense(15,
				kernel_initializer=self.initializer, use_bias=False, activation="tanh")(x)
			
			self.models_action.append(keras.Model(
				inputs=self.model_action_i_state,
				outputs=self.model_action_o_action,
				name="model_action_{}".format(i)))
			#self.model_action.summary()


	def create_reward_model(self):
		self.model_reward_i_state = keras.Input(shape=(self.state_size))
		self.model_reward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_reward_i_state, self.model_reward_i_action])
		x = self.module_dense(x, self.state_size, n2=self.state_size)

		self.model_reward_o_reward_step = layers.Dense(1,
			kernel_initializer=self.initializer, use_bias=False)(x)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=[self.model_reward_o_reward_step],
			name="model_reward")
		#self.model_reward.summary()
	

	# predict encoding of next image from state and action
	def create_encoding_model(self):
		self.model_encoding_i_state = keras.Input(shape=(self.state_size))
		self.model_encoding_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_encoding_i_state, self.model_encoding_i_action])
		x = self.module_dense(x, self.state_size, n2=self.state_size)

		self.model_encoding_o_image_enc = layers.Dense(self.image_enc_size,
			kernel_initializer=self.initializer, use_bias=False, activation="tanh")(x)

		self.model_encoding = keras.Model(
			inputs=[self.model_encoding_i_state, self.model_encoding_i_action],
			outputs=[self.model_encoding_o_image_enc],
			name="model_encoding")
		#self.model_encoding.summary()


	def advance(self, image, action_prev):
		image = tf.convert_to_tensor(image, dtype=tf.float32) * 0.0039215686274509803 # 1/255
		action_prev = tf.expand_dims(convert_action_to_continuous(action_prev),0)
		image_enc = self.model_image_encoder(tf.expand_dims(image, 0), training=False)
		image_enc_pred = self.model_encoding([self.state, action_prev], training=False)

		self.state = self.model_state([self.state, image_enc], training=False)

		return tf.reduce_mean(tf.square(image_enc[0] - image_enc_pred[0]))

	"""
	Reset state (after an episode)
	"""
	def reset_state(self):
		self.state = tf.zeros((1, self.state_size))

	"""
	Predict action from the state of the model
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self, model_id, epsilon=0.0):
		state_input = (1.0-epsilon)*self.state +\
			epsilon*tf.random.uniform((1, self.state_size), -1.0, 1.0)
		action = self.models_action[model_id](state_input, training=False)[0]

		return convert_action_to_mixed(action)

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
			memory.compute_states(self.model_state, self.model_image_encoder)
			
			images, actions, rewards, state_init = memory.get_sample(self.replay_sample_length)

			# train the image encodet model (and reward model, 1st phase)
			state_prev = state_init
			for i in range(self.replay_sample_length-self.tbptt_length_encoder):
				state_prev = self.train_image_encoder_model(images, actions, rewards, state_prev,
					tf.convert_to_tensor(i))
				print("Epoch {:3d} - Training image encoder model ({}/{})".format(
					e, i+self.tbptt_length_encoder+1, self.replay_sample_length), end="\r")
			print("")

			for i in range(self.replay_sample_length):
				image_encs[i].assign(self.model_image_encoder(images[i], training=False))
				print("Computing image encodings... {} / {}      ".format(i,
				self.replay_sample_length), end="\r")

			# train the backbone (image encoding, state and reward models)
			state_prev = state_init
			for i in range(self.replay_sample_length-self.tbptt_length_backbone):
				state_prev = self.train_backbone(image_encs, actions, rewards, state_prev,
					tf.convert_to_tensor(i))
				print("Epoch {:3d} - Training the backbone ({}/{})".format(
					e, i+self.tbptt_length_backbone+1, self.replay_sample_length), end="\r")
			print("")
			
			# train the action (policy) models
			for j in range(self.n_replay_episodes):
				self.active_action_model = j
				state_prev = state_init
				for i in range(self.replay_sample_length):
					state_prev = self.train_action_model(image_encs, actions, rewards, state_prev,
						tf.convert_to_tensor(i))
					print("Epoch {:3d} - Training action model {} ({}/{})".format(
						e, j, i+1, self.replay_sample_length), end="\r")
				print("")
			
			self.save_model("model/model")

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


	def save_model(self, filename_prefix):
		print("Saving model with prefix: {}".format(filename_prefix))
		self.model_image_encoder.save_weights("{}_image_encoder.h5".format(filename_prefix))
		self.model_image_decoder.save_weights("{}_image_decoder.h5".format(filename_prefix))
		self.model_state.save_weights("{}_state.h5".format(filename_prefix))
		for i in range(self.n_replay_episodes):
			self.models_action[i].save_weights("{}_action_{}.h5".format(filename_prefix, i))
		self.model_reward.save_weights("{}_reward.h5".format(filename_prefix))
		self.model_encoding.save_weights("{}_encoding.h5".format(filename_prefix))
	
	def load_model(self, filename_prefix):
		print("Loading model with prefix: {}".format(filename_prefix))
		self.model_image_encoder.load_weights("{}_image_encoder.h5".format(filename_prefix))
		self.model_image_decoder.load_weights("{}_image_decoder.h5".format(filename_prefix))
		self.model_state.load_weights("{}_state.h5".format(filename_prefix))
		for i in range(self.n_replay_episodes):
			self.models_action[i].load_weights("{}_action_{}.h5".format(filename_prefix, i))
		self.model_reward.load_weights("{}_reward.h5".format(filename_prefix))
		self.model_encoding.load_weights("{}_encoding.h5".format(filename_prefix))

		#self.create_recurrent_module()