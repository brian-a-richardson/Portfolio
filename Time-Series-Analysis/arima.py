"""
ARIMA
Date: 7/15/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import load_csv_files
import joblib
import os
import csv
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


'''
Functions
'''

def train_sarimax_model(df, target_col, exog_cols=None, order=(3, 3, 3), seasonal_order=(3, 3, 3, 5)):
    exog = df[exog_cols] if exog_cols is not None else None
    model = SARIMAX(df[target_col], exog=exog, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    mape = np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100
    return mape

def evaluate_model(models, test_name, df, target_col, exog_cols=None):
	# Calculate, print, and save the error metrics.
	exog = df[exog_cols] if exog_cols is not None else None

	for name, model in models.items():
		predictions = model.forecast(steps=len(df), exog=exog)
		actuals = df[target_col].iloc[-len(predictions):]

		mse = mean_squared_error(actuals, predictions)
		mae = mean_absolute_error(actuals, predictions)
		rmse = np.sqrt(mse)
		mape = calculate_mape(actuals, predictions)

		# Display the errors
		print(f'Mean Squared Error for {df}: {mse:.4f}')
		print(f'Mean Absolute Error for {df}: {mae:.4f}')
		print(f'Root Mean Squared Error for {df}: {rmse:.4f}')
		print(f'Absolute Percentage Error for {df}: {mape:.4f}')

		# Save the results 
		save_results(test_name, name, mse, mae, rmse, mape)


def save_results(test_name, model_name, mse, mae, rmse, mape):
	# Create folder if one doesn't exist
	folder_path = "Results"
	if not os.path.exists(folder_path):
		os.makedirs(folder_path) 

	# Create the results file name
	filename = "ARIMA Results.csv"
	file_path = os.path.join(folder_path, filename)

	# Create a time stamp
	now = datetime.now()
	timestamp = now.strftime('%Y-%m-%d_%H-%M-%S') # Format the data and time

	# Create a dataframe name
	df_name = name.replace('.csv', '')

	# Create a the row data
	data = {
		'Name': test_name,
		'Model': model_name,
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
		train_data_frames, number_of_df, ex_time = load_csv_files(train_folder_path, 20, False)
		
		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into training data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

		valid_data_frames, number_of_df, ex_time = load_csv_files(valid_folder_path, 5, False)

		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into validation data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	# Set SARIMAX order (p, d, q)
	sarimax_order = (1, 1, 1)
	exogenous_vars = ['close', 'volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
							'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
							'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
							'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
							'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
							'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
							'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close'   ]
	models = {}
	target_col = 'close'


	# Train the model
	for name, df in train_data_frames.items():
		print(f"Processing DataFrame: {name}")

		# Fit the model
		model_fit = train_sarimax_model(df, target_col, exogenous_vars)
		models[name] = model_fit
		print(f"{name} model has been trained")

	# Evaluate the model on test data
	for name, df in valid_data_frames.items():
		evaluate_model(models, name, df, target_col, exogenous_vars)

