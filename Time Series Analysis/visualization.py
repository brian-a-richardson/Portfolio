"""
Data Visualization Script
Date: 7/8/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import os


from functions import *

if __name__ == "__main__":
	# Specify the folder path
	folder_path = "Cleaned Data/Regression/"
	try:
		# Load all csv files
		data_frames, number_of_df, ex_time = load_csv_files(folder_path, 100, False)

		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	# Create a scatter plot
	res = list(data_frames.keys())[0]

	# scatter_plot(data_frames[res])
	
	# line_plot(data_frames[res])

	# auto_plot(data_frames[res])

	# Find min and max values of closing price
	min_value, max_value = find_min_max(data_frames, 'close')
	print(f"Min: {min_value} Max: {max_value}") 


