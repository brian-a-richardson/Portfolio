"""
Data Exploration Script
Date: 6/11/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import os
import pandas as pd

from IPython.display import display

# path to the data folder
folder_path = 'Data/'

# get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# create an empty dictionary to store DataFrames
data_frames = {}

# loop through each CSV file and read it into a DataFrame
for file in csv_files:
	file_path = os.path.join(folder_path, file)
	df_name = os.path.splitext(file)[0] # Use the file name (without extension) as DataFrame name
	data_frames[df_name] = pd.read_csv(file_path)

# New Dictionary to hold merged dataframes
merged_data_frames = {}

# Loop through piars of DataFrames in the dictionary
keys = list(data_frames.keys())
for i in range(0, len(keys), 2):
	df1_key = keys[i]
	df2_key = keys[i + 1] if i + 1 <len(keys) else None

	if df2_key:
		df1 = data_frames[df1_key]
		df2 = data_frames[df2_key]
		merged_df_key = df2_key
		merged_data_frames[merged_df_key] = pd.merge(df1, df2, on="Date", how="inner")

# create a variable to keep track of total number of rows
total_rows = 0

# check each DataFrame
for key, df in merged_data_frames.items():
	print(f"Merged DataFrame for {key}:")
	display(df.head(10))

	# compute the number of rows
	rows = len(df.axes[0])
	total_rows += rows
	print("Number of rows: ", rows)

	print()

print()
print("The total number of rows is: ", total_rows)