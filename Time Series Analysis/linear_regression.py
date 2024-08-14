"""
Linear Regression
Date: 7/15/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import load_csv_files
import joblib
import os
import csv

'''
Functions
'''
def prepare_data(data, target_column):
	# Split the data inot features and target variables
	X = data.drop(columns=[target_column])
	y = data[target_column]
	return X, y

def linear_regression(train_data, valid_data, target_column):
	# Initialize linear regression model
	model = SGDRegressor(loss='squared_epsilon_insensitive')

	for df in train_data:
		print(f'Building a model for {df}')

		# Prepare features and target variable
		train_X, train_y = prepare_data(train_data[df], target_column)
		
		# Fit the model to the training data
		model.partial_fit(train_X, train_y)
		print(f'Model trained on {df}')

	for df in valid_data:
		print(f'Testing data on {df}')

		# Prepare validation data for testing
		valid_X, valid_y = prepare_data(valid_data[df], target_column)

		# Make predictions 
		y_pred = model.predict(valid_X)

		# Calculate the errors
		mse = round(mean_squared_error(valid_y, y_pred), 5)
		mae = round(mean_absolute_error(valid_y, y_pred), 5)
		rmse = round(np.sqrt(mse), 5)
		mape = round(np.mean(np.abs((valid_y - y_pred) / valid_y)) * 100, 5)

		# Display the errors
		print(f'Mean Squared Error for {df}: {mse:.2f}')
		print(f'Mean Absolute Error for {df}: {mae:.2f}')
		print(f'Root Mean Squared Error for {df}: {rmse:.2f}')
		print(f'Absolute Percentage Error for {df}: {mape:.2f}')

		# Save the results 
		save_results(df, mse, mae, rmse, mape)

	# Save the trained model
	joblib.dump(model, 'linear_regression_model.pkl')
	print('Model save as linear_regression_model.pkl')


def save_results(name, mse, mae, rmse, mape):
	# Create folder if one doesn't exist
	folder_path = "Results"
	if not os.path.exists(folder_path):
		os.makedirs(folder_path) 

	# Create the results file name
	filename = "Linear Regression Results.csv"
	file_path = os.path.join(folder_path, filename)

	# Create a time stamp
	now = datetime.now()
	timestamp = now.strftime('%Y-%m-%d_%H-%M-%S') # Format the data and time

	# Create a dataframe name
	df_name = name.replace('.csv', '')

	# Create a the row data
	data = {
		'Name': df_name,
		'Time': timestamp,
		'MSE': mse,
		'MAE': mae,
		'RMSE': rmse,
		'MAPE': mape
	}

	add_row_to_csv(file_path, list(data.values()), list(data.keys()))

def add_row_to_csv(file_path, row, header):
    """Add a row to a CSV file. Create the file if it does not exist."""
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(header)  # Write header

        # Write the new row
        writer.writerow(row)
        print(f'Added row: {row} to {file_path}')


'''
Main
'''

if __name__ == '__main__':
	# Specify the folder path
	train_folder_path = "Cleaned Data/Regression/Train"
	valid_folder_path = "Cleaned Data/Regression/Validation"
	try:
		# Load all csv files
		train_data_frames, number_of_df, ex_time = load_csv_files(train_folder_path, 60, False)
		
		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into training data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

		valid_data_frames, number_of_df, ex_time = load_csv_files(valid_folder_path, 20, False)

		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into validation data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	# Create the model from the data
	linear_regression(train_data_frames, valid_data_frames, "close")

