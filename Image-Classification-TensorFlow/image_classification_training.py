"""
Image Classification Model Training Script
Author:  Brian Richardson
Last Updated:  9/12/2024
Description:  A script to train models to classify images using TesorFlow.
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os
import json

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''
Functions
'''
# Set up the variables
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
file_name = 'flower_photos.tar'
destination_dir = "data"
cache_dir = pathlib.Path('.')
data_dir = cache_dir / destination_dir
model_destination = 'models'
model_dir = cache_dir / model_destination

def load_data(url, file_name, destination_dir, chache_dir, data_dir):
	# Use get_file to download and extract the dataset
	file_path = tf.keras.utils.get_file(
		file_name, 						# The name of the file to be saved
		origin=dataset_url, 			# The URL to download the file from 
		extract=True, 					# Whether to extract the file
		cache_dir=str(cache_dir), 					# Set chache_dir to the current working directory
		cache_subdir=destination_dir	# Specify the subdirectory under cache_dir
		)
	# Convert the path to a pathlib.Path object
	data_dir = pathlib.Path(file_path).with_suffix('')

	return data_dir

def classes_to_json(class_names):
	json_file_path = 'class_names.json'
	try:
		with open(json_file_path, 'w') as json_file:
			json.dump(class_names, json_file, indent=4) # Indent parameter for pretty printing
		print(f"Class names successfully saved to {json_file_path}")
	except IOError as e:
		print(f"Error saving data to {json_file_path}: {e}") 

def analyze_data(data_dir):
	# Count the images
	image_count = len(list(data_dir.rglob('*/*.jpg')))
	print(f"There are {image_count} images in the data set")

def train_model(data_dir, model_dir, model_name="default_model.keras", epochs=10):
	# Define the parameters
	batch_size = 32
	img_height = 180
	img_width = 180

	# set up training data set
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split = 0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
		)

	# set up validation data set
	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
		)

	class_names = train_ds.class_names
	print(f"Class Names: {class_names}")

	# Visualize the data
	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")
	plt.show()

	# Configure the dataset for performance
	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# Data Augmentation Layer
	data_augmentation = Sequential([
		layers.RandomFlip("horizontal", 
							input_shape=(img_height, img_width, 3)),
		layers.RandomRotation(0.1),
		layers.RandomZoom(0.1)
		])

	# Visualize Augmentation
	plt.figure(figsize=(10, 10))
	for images, _ in train_ds.take(1):
		for i in range(9):
			augmented_images = data_augmentation(images)
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(augmented_images[0].numpy().astype("uint8"))
			plt.axis("off")


	# Create a model path and check if the model exists already.  If it does load the model.
	# If it does not create a new model.
	model_path = model_dir / model_name
	if  os.path.exists(model_path):
		print(f"Loading existing model form {model_path}")
		model = tf.keras.models.load_model(model_path)
	else:
		print(f"Model not found.  Creating and training a new model...")
		# Create the model
		num_classes = len(class_names)

		model = Sequential([
			layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
			data_augmentation,
			layers.Conv2D(16, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Conv2D(32, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Conv2D(64, 3, padding='same', activation='relu'),
			layers.MaxPooling2D(),
			layers.Dropout(0.2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(num_classes, name="outputs")
			])

	# Compile the model
	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	# View model summary
	model.summary()

	# Train the model
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs)

	# Save the model
	model.save(model_path)
	print(f"Model saved to {model_path}")

	# Save class names to JSON array
	classes_to_json(class_names)

	# Visualize the training results
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8,8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label="Training Accuracy")
	plt.plot(epochs_range, val_acc, label="Validation Accuracy")
	plt.legend(loc='lower right')
	plt.title("Training and Validation Accuracy")

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()

if __name__ == '__main__':
	# Check if the directory already exists
	if not data_dir.exists() or not any(data_dir.iterdir()):  # The directory does not exist or is empty
		data_dir = load_data(dataset_url, file_name, destination_dir, cache_dir, data_dir)
		print(f"Data set downloaded and saved to: {data_dir}")
	else:
		data_dir = data_dir / 'flower_photos'
		print(f"Data already exists at: {data_dir}")

	analyze_data(data_dir)

	train_model(data_dir, model_dir, "flower_model.keras", 20)