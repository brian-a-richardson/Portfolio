"""
Image Classification Classify Script
Author:  Brian Richardson
Last Updated:  9/12/2024
Description:  This script loads keras models and uses them to classify images of flowers.
"""

import tensorflow as tf
import numpy as np
import os
import json

# Set up the variables
img_height = 180
img_width = 180
model_dir = "models/flower_model.keras"
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file("Red_sunflower", origin=sunflower_url)
class_names_path = "class_names.json"

'''
Functions
'''

def load_model(model_path):
	if os.path.exists(model_path):
		print(f"Loading model form {model_path}")
		model = tf.keras.models.load_model(model_path)
		return model
	else: 
		raise FileNotFoundError(f"Model file not found at {model_path}")

def get_image(img_path):
	print(f"Loading image from {img_path}")
	image = tf.keras.utils.load_img(
		img_path, target_size=(img_height, img_width)
		)
	img_array = tf.keras.utils.img_to_array(image)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	return img_array

def get_class_names(class_names_path):
	try:
		with open(class_names_path, 'r') as f:
			class_names = json.load(f)
			print(f"Class names: {class_names}")
		return class_names
	except FileNotFoundError:
		print(f"Error:  The file {class_names_path} was not found.")
		return None
	except json.JSONDecodeError:
		print(f"Error:  The file {class_names_path} is not a valid JSON file.")
		return None
	except Exception as e:
		print(f"An unexpected error occured.  {e}")
		return None

def main():
	model = load_model(model_dir)
	img = get_image(sunflower_path)
	class_names = get_class_names(class_names_path)

	predictions = model.predict(img)
	score = tf.nn.softmax(predictions[0])

	print("This image most likely belongs to {} with a {:.2f} percent confidence."
		.format(class_names[np.argmax(score)], 100 * np.max(score))
		)

if __name__ == '__main__':
	main()
