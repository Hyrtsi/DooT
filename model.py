import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers


class Model:
	def __init__(self):
		self.initializer = initializers.RandomNormal(stddev=0.04)
		self.create_model()

	def create_model(self):
		inputs = keras.Input(shape=(240, 320, 3))
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

		self.model = keras.Model(inputs=inputs, outputs=outputs)
		self.model.summary()

		self.model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam()
		)

	def train_batch(self, x, y):
		self.model.train_on_batch(x, y)

	def predict_action(self, x):
		prediction = self.model.predict(x)[0]
		action = np.where(prediction > 0.0, True, False).tolist()
		action[14] = prediction[14]*100.0
		#print(action)
		return action