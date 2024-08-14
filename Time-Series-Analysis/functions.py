"""
Functions to aid with the data analysis
Date: 7/1/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import time
import os
import random
import shutil 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

"""
Load all csv to pandas data frames
"""

def load_csv_files(folder_path, num_files, shuffle=True):
	# Get the function start time
	start = time.time()

	# Set up a counter for how many files are loaded
	files_loaded = 0

	# Check if the folder path exists
	if not os.path.exists(folder_path):
		raise FileNotFoundError(f"The directory '{folder_path}' does not exist")

	# Get all files in the folder
	files = os.listdir(folder_path)

	# Filter out only CSV files
	csv_files = [file for file in files if file.endswith('.csv')]

	# Raise error if no CSV files found
	if not csv_files:
		raise ValueError(f"No valid CSV files found in '{folder_path}'")

	# Randomly shuffle the list of files to get a random sample
	if shuffle:
		random.shuffle(csv_files)

	# Initialize an empty dictionary to store data frames
	dfs = {}

	# Load each CSV file into a pandas data frame
	for file in csv_files:
		if len(dfs) == num_files:
			break
		try:
			file_path = os.path.join(folder_path, file)
			df = pd.read_csv(file_path)
			files_loaded += 1
			dfs[file] = df
			print(f"Loaded {file} into DataFrame")
		except Exception as e:
			print(f"Error loading {file}: {e}")

	# Calculate the time it took to exectue
	end = time.time()
	execution_time = round(end - start, 2)

	return dfs, files_loaded, execution_time

'''
Data Frame Cleaning 
'''

def clean_data_frame(data_frame, date=False):
	# Get the function start time and start rows
	start = time.time()

	# Count the attributes
	start_rows = len(data_frame)
	start_cols = len(data_frame.columns)

	# Create a cleaned data frame
	cleaned_df = data_frame

	# Calculate the rolling average and add a new column
	cleaned_df['ra_open'] = data_frame['open'].rolling(window=len(data_frame), min_periods=1).mean().round(2)
	cleaned_df['ra_high'] = data_frame['high'].rolling(window=len(data_frame), min_periods=1).mean().round(2)
	cleaned_df['ra_low'] = data_frame['low'].rolling(window=len(data_frame), min_periods=1).mean().round(2)
	cleaned_df['ra_close'] = data_frame['close'].rolling(window=len(data_frame), min_periods=1).mean().round(2)

	# Drop rows that have any missing values
	cleaned_df = data_frame.dropna()

	# Drop the columns that are not needed and keep only the ones that are useful
	regression_columns_to_keep = ['close', 'volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
							'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
							'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
							'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
							'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
							'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
							'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close'   ]
	regression_cleaned_df = cleaned_df[regression_columns_to_keep] 

	timeseries_columns_to_keep = ['datetime', 'close', 'volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
							'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
							'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
							'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
							'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
							'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
							'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close'   ]
	timeseries_cleaned_df = cleaned_df[timeseries_columns_to_keep] 

	# Treate outliers
	# cleaned_df = replace_local_outliers_with_local_average(cleaned_df, window_size=20, threshold=0.1)

	# Count remaining rows and get the df name
	rows = len(regression_cleaned_df)
	dp_rows = start_rows - rows
	cols = len(regression_cleaned_df.columns)
	dp_cols = start_cols - cols

	# Calculate the time it took to execute
	end = time.time()
	execution_time = round(end - start, 2)

	return regression_cleaned_df, timeseries_cleaned_df, execution_time, rows, dp_rows, cols, dp_cols

'''
Splitting the Data
'''

def split_data(source_folder, train, validation, test):
	# Get the function start time
	start = time.time()

	# Check if the folder path exists
	if not os.path.exists(source_folder):
		raise FileNotFoundError(f"The directory '{source_folder}' does not exist")

	# Check if the inputs are valid	
	if train + validation + test != 1:
		raise ValueError(f"The train, validation, and test values must add up to 1.")

	# Set up the folder paths
	folder1 = source_folder + "Train"
	folder2 = source_folder + "Validation"
	folder3 = source_folder + "Test"

	# Create the destination folders if they don't exist
	os.makedirs(folder1, exist_ok=True)
	os.makedirs(folder2, exist_ok=True)
	os.makedirs(folder3, exist_ok=True)

	# Get the list of files in source folder
	files = os.listdir(source_folder)

	# Filter out only CSV files
	csv_files = [file for file in files if file.endswith('.csv')]
	number_of_files = len(csv_files)

	# Raise error if no CSV files found
	if not csv_files:
		raise ValueError(f"No CSV files found in '{folder_path}'")

	# Calculate the number of files for each destination folder
	train_files = ceil(number_of_files * train)
	validation_files = ceil(number_of_files * validation)
	test_files = number_of_files - train_files - validation_files # Remaining files

	# Randomly shuffle the list of files
	random.shuffle(csv_files)

	# Copy files to destination folder
	for i, file in enumerate(csv_files):
		source_file =  os.path.join(source_folder, file)
		if i < train_files:
			shutil.copy(source_file, folder1)
		elif i < train_files +validation_files:
			shutil.copy(source_file, folder2)
		else:
			shutil.copy(source_file, folder3)

	print(f"The data has been split into train, validation, and test sets.")
	print(f"Files distributed into folders:")
	print(f"Train: {train_files}")
	print(f"Validation: {validation_files}")
	print(f"Test: {test_files}")

'''
Clean local outliers
'''

def replace_local_outliers_with_local_average(data, window_size=20, threshold=1):
    """
    Function to replace local outliers in a DataFrame with local average using scikit-learn.
    
    Parameters:
    - data: A pandas DataFrame containing numeric columns.
    - window_size: Size of the sliding window to calculate local average.
    - threshold: The contamination factor for LOF (proportion of outliers expected).
    
    Returns:
    - cleaned_data: A pandas DataFrame with local outliers replaced by local average.
    """
    cleaned_data = data.copy()
    columns = data.columns
    
    for col in columns:
        # Compute local outlier factor
        lof = LocalOutlierFactor(n_neighbors=window_size, contamination=threshold)
        outliers = lof.fit_predict(data[[col]])
        
        # Replace outliers with local average
        for i, is_outlier in enumerate(outliers):
            if is_outlier == -1:  # -1 indicates an outlier
                start_idx = max(0, i - window_size)
                end_idx = min(len(data), i + window_size + 1)
                local_avg = data[col].iloc[start_idx:end_idx].mean()
                cleaned_data.loc[i, col] = local_avg
    
    return cleaned_data

'''
Scatter Plot
'''

def scatter_plot(input_data):
	# Specify the dependent in independent variables
	columns = input_data.columns
	x_var = columns[0]
	columns.delete(0)

	# Create several scatter plots using x_var as the dependent variable and all other variables as independent variables
	for i in range(len(columns)):
		y_var = columns[i]

		# Create scatter plot using seaborn
		plt.figure(figsize=(8,6))
		sns.scatterplot(x=x_var, y=y_var, data=input_data)
		plt.title(f"Scatter Plot: {y_var} vs {x_var}")
		plt.xlabel(x_var)
		plt.ylabel(y_var)
		plt.grid(True)
		plt.show()


'''
Line Plot
'''

def line_plot(input_data):
	# Specify the dependent in independent variables
	columns = input_data.columns

	# Create line plots using for each variable
	for i in range(len(columns)):
		# Create line plot using Seaborn
		plt.figure(figsize=(8, 6))
		sns.lineplot(data=input_data[columns[i]])
		plt.title(f"Line Plot of {columns[i]}")
		plt.xlabel('Index')
		plt.ylabel(columns[i])
		plt.grid(True)
		plt.show()

'''
Autocorrelation Plot
'''

def auto_plot(input_data):
	# Specify the dependent in independent variables
	columns = input_data.columns
	x_var = columns[0]
	columns.delete(0)


	# Create several auto correlation plots
	for i in range(len(columns)):
		# Calculate autocrrelation
		autocorr_values = input_data[columns[i]].autocorr()

		# Create autocorrelation plot using Seaborn
		plt.figure(figsize=(10, 6))
		plt.acorr(input_data[columns[i]], maxlags=len(input_data)-1)
		plt.title(f'Autocorrelation Plot of {columns[i]} (Autocorr = {autocorr_values:.2f})')
		plt.xlabel('Lag')
		plt.ylabel('Autocorrelation')
		plt.grid(True)
		plt.show()

def find_min_max(dataframes, column_name):
    """Find the minimum and maximum values of a specific column across multiple DataFrames."""
    # Initialize min and max variables
    overall_min = float('inf')  # Start with a large number for min
    overall_max = float('-inf') # Start with a small number for max

    # Loop through each DataFrame
    for df in dataframes.values():
        if column_name in df.columns:  # Check if column exists
            current_min = df[column_name].min()
            current_max = df[column_name].max()

            overall_min = min(overall_min, current_min)
            overall_max = max(overall_max, current_max)
        else:
            print(f"Column '{column_name}' not found in one of the DataFrames.")

    return overall_min, overall_max