"""
Data Preparation and Cleaning Script
Date: 7/1/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import os

from functions import *

if __name__ == "__main__":
	# Specify the folder path
	folder_path = "Raw Data"
	try:
		# Load all csv files
		data_frames, number_of_df, ex_time = load_csv_files(folder_path, 100, True)

		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	# Set up counters
	total_time = 0
	total_rows = 0
	total_dp_rows = 0

	# Set up dictionaries
	regression_data_frames = {}
	time_series_data_frames = {}

	# Clean the data
	for key, df in data_frames.items():
		regression_data_frames[key], time_series_data_frames[key], ex_time, rows, dp_rows, cols, dp_cols = clean_data_frame(data_frames[key])


		# Calcualte totals
		total_time += ex_time
		total_rows += rows
		total_dp_rows += dp_rows

		# Display task completion message
		print(f"'{key}' data frame has been cleaned.  Elapsed time: '{ex_time}'")

	# Display cleaning complete message
	print()
	print(f"The cleaning is complete.")
	print(f"It took '{total_time}' seconds, and resulted in '{len(data_frames)}' data frames with '{total_rows}' total rows. '{total_dp_rows}' have been dropped.")
	print(f"{dp_cols} columns were dropped resulting in '{cols}' columns.")

	# Set the destination
	destination = './Cleaned Data/Regression/'

	# Create directory if on doesn't exist
	os.makedirs(destination, exist_ok=True)

	# Convert data frames back to csv files and put them in the 'Cleaned Data/Regression/' folder
	for key, df in regression_data_frames.items():
		# Define File Path
		file_path = destination + f"{key}"

		# Convert the DataFrame to CSV
		df.to_csv(file_path, index=False)

		# Print results
		print(f"DataFrame '{key}' successfullly saved as {file_path}")

	# Sort the cleaned folder into train, validation, and test data
	try:
		split_data(destination, 0.6, 0.2, 0.2)
	except FileNotFoundError as e:
		print(e)
	except ValueError as e:
		print(e)

	# Set the destination
	destination = './Cleaned Data/Time Series/'

	# Create directory if on doesn't exist
	os.makedirs(destination, exist_ok=True)

	# Convert data frames back to csv files and put them in the 'Cleaned Data/Time Series/' folder
	for key, df in time_series_data_frames.items():
		# Define File Path
		file_path = destination + f"{key}"

		# Convert the DataFrame to CSV
		df.to_csv(file_path, index=False)

		# Print results
		print(f"DataFrame '{key}' successfullly saved as {file_path}")

	# Sort the cleaned folder into train, validation, and test data
	try:
		split_data(destination, 0.6, 0.2, 0.2)
	except FileNotFoundError as e:
		print(e)
	except ValueError as e:
		print(e)