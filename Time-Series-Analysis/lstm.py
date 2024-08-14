"""
LSTM
Date: 7/15/2024
Author:  Brian Richardson
Class:  DATA 670
"""

'''
Execute the linux command line with wsl.exe
Tensorflow is not supported for Windows
'''

import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from functions import load_csv_files
import glob
import joblib
import os
import csv

'''
Functions
'''

def preprocess_dataframe(df, target_col, exog_cols):
    """Scale features and target variable."""
    scaler = MinMaxScaler()
    df[exog_cols] = scaler.fit_transform(df[exog_cols])
    df[target_col] = scaler.fit_transform(df[[target_col]])
    return df, scaler

def create_lstm_model(input_shape):
    """Create and compile an LSTM model."""
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))  # Assuming we want to predict one target value
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df, target_col, exog_cols, time_steps=1):
    """Prepare data for LSTM."""
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df[exog_cols].iloc[i:i + time_steps].values)
        y.append(df[target_col].iloc[i + time_steps])
    return np.array(X), np.array(y)

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    mape = np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100
    return mape

def evaluate_model(model, test_name, X_test, y_test, scaler):
    """Evaluate the model and print error metrics."""
    for name, model in models.items():
	    predictions = model.predict(X_test)
	    predictions = scaler.inverse_transform(predictions)  # Inverse scale predictions
	    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

	    mse = mean_squared_error(y_test, predictions)
	    mae = mean_absolute_error(y_test, predictions)
	    rmse = np.sqrt(mse)
	    mape = calculate_mape(y_test, predictions)

	    # Display Errors
	    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%')
	    
	    # Save the results
	    save_results(test_name, name, mse, mae, rmse, mape)

def save_results(test_name, model_name, mse, mae, rmse, mape):
	# Create folder if one doesn't exist
	folder_path = "Results"
	if not os.path.exists(folder_path):
		os.makedirs(folder_path) 

	# Create the results file name
	filename = "LSTM Results.csv"
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

	models = {}
	scalers = {}
	target_col = "close"
	exog_cols = ['volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
							'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
							'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
							'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
							'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
							'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
							'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close'   ]
	time_steps = 3  # Number of time steps for LSTM

	for name, df in train_data_frames.items():
		print(f'Started training model using {name}')
		df, scaler = preprocess_dataframe(df, target_col, exog_cols)
		X_train, y_train = prepare_data(df, target_col, exog_cols, time_steps)

		model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
		model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

		models[name] = model # Save model
		scalers[name] = scaler # Save scalers
		print(f'Model trained for {name}')

	# Evalute models on test data
	for  name, df in valid_data_frames.items():
		df, scaler = preprocess_dataframe(df, target_col, exog_cols)
		X_test, y_test = prepare_data(df, target_col, exog_cols, time_steps)
		evaluate_model(models, name, X_test, y_test, scaler)
