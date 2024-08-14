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
import tensorflow as tf
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from functions import load_csv_files
import glob
import joblib
import os
import csv

'''
Functions
'''

def preprocess_data(df, sequence_length, target_column, exog_columns):
    # Assuming the target column is named 'target', adjust as per your dataset
    target = df[target_column].values
    
    # Normalize time series features using standard scaler
    scaler_ts = StandardScaler()
    features_ts = scaler_ts.fit_transform(df.drop(columns=[target_column] + exog_columns).values)
    
    # Normalize exogenous columns separately using standard scaler
    scaler_exog = StandardScaler()
    features_exog = scaler_exog.fit_transform(df[exog_columns].values)
    
    # Prepare sequences of data and targets
    X_ts, X_exog, y = [], [], []
    for i in range(len(features_ts) - sequence_length):
        X_ts.append(features_ts[i:i+sequence_length])
        X_exog.append(features_exog[i+sequence_length-1])  # Take the last value of exog for each sequence
        y.append(target[i+sequence_length])
    
    X_ts = np.array(X_ts)
    X_exog = np.array(X_exog)
    y = np.array(y)
    
    return X_ts, X_exog, y, scaler_ts, scaler_exog

def create_lstm_model(sequence_length, ts_input_shape, exog_input_shape):
    # LSTM model architecture
    input_ts = Input(shape=(sequence_length, ts_input_shape), name='input_ts')
    input_exog = Input(shape=(exog_input_shape,), name='input_exog')
    
    lstm_output = LSTM(50, activation='tanh')(input_ts)
    concatenated = Concatenate()([lstm_output, input_exog])
    output = Dense(1)(concatenated)
    
    model = tf.keras.Model(inputs=[input_ts, input_exog], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(name, df, sequence_length, batch_size, epochs):
	# Define exogenous columns (adjust as per your dataset)
    exog_columns = ['volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
						'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
						'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
						'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
						'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
						'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
						'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close']  # Add more columns as needed
    
    # Preprocess data
    X_ts, X_exog, y, scaler_ts, scaler_exog = preprocess_data(df, sequence_length=sequence_length, target_column='close', exog_columns=exog_columns)
    
    # Split data into training and validation sets
    X_ts_train, X_ts_val, X_exog_train, X_exog_val, y_train, y_val = train_test_split(X_ts, X_exog, y, test_size=None)

    model_path = f"LSTM/lstm_model.h5"
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = load_model(model_path)
    else:
        print("Creating new model.")        
        
        # Create LSTM model
        model = create_lstm_model(sequence_length, X_ts.shape[2], X_exog.shape[1])
    
    # Define callbacks (save the best model based on validation loss)
    checkpoint_path = f"LSTM/{name}_model_checkpoint.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit({'input_ts': X_ts_train, 'input_exog': X_exog_train}, y_train, 
                        epochs=epochs, batch_size=batch_size,
                        validation_data=({'input_ts': X_ts_val, 'input_exog': X_exog_val}, y_val),
                        callbacks=[checkpoint, early_stopping])
    
    # Evaluate the best model on validation data
    model = tf.keras.models.load_model(checkpoint_path)  # Load the best model saved during training
    val_loss = model.evaluate({'input_ts': X_ts_val, 'input_exog': X_exog_val}, y_val)
    print(f"Validation Loss: {val_loss}")
    
    # Optionally, save the scalers for later use
    if not os.path.exists(model_path):
        joblib.dump(scaler_ts, f"scaler_ts.pkl")
        joblib.dump(scaler_exog, f"scaler_exog.pkl")
    
    # Save the final trained model
    model.save(model_path)
    print(f"Model saved for {name}.")

def test_lstm_model(name, df, sequence_length): 
    model_path = f"LSTM/lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
        
        # Define exogenous columns (adjust as per your dataset)
        exog_columns = ['volume', 'bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
							'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100', 'ad_0', 'adosc_100_200', 'obv_0', 'atr_100', 'natr_100', 'trange_0', 'avgprice_0', 'medprice_0',
							'typprice_0', 'wclprice_0', 'ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0', 
							'adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
							'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
							'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
							'trix_100', 'ultosc_100_200_300', 'willr_100', 'ra_open', 'ra_high', 'ra_low', 'ra_close'   ]  # Add more columns as needed
        
        # Preprocess data for testing
        X_ts, X_exog, y, scaler_ts, scaler_exog = preprocess_data(df, sequence_length=sequence_length, target_column='close', exog_columns=exog_columns)
        
        # Predict on test data
        y_pred = model.predict({'input_ts': X_ts, 'input_exog': X_exog})
        
        # Evalute Model
        evaluate_model(name, y, y_pred)
    else:
        print(f"Model {model_path} not found. Skipping testing.")

def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    actual, forecast = np.array(actual), np.array(forecast)
    non_zero_actual = actual != 0
    mape = np.mean(np.abs((actual[non_zero_actual] - forecast[non_zero_actual]) / actual[non_zero_actual])) * 100
    return mape

def evaluate_model(test_name, y_val, y_pred):
		mse = mean_squared_error(y_val, y_pred)
		mae = mean_absolute_error(y_val, y_pred)
		rmse = np.sqrt(mse)
		mape = calculate_mape(y_val, y_pred)

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

def date_string_to_float(date_str):
	# Convert date string to datetime object
	date_obj = datetime.strptime(date_str, '%Y-%m-%d')

	# Convert datetime object to Unix timestamp(float)
	timestamp = datetime.timestamp(date_obj)

	return timestamp

'''
Main
'''

if __name__ == '__main__':
	# Specify the folder path
	train_folder_path = "Cleaned Data/Time Series/Train"
	valid_folder_path = "Cleaned Data/Time Series/Validation"
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

		# Change datetime format
		for df in train_data_frames.values():
			df['datetime'] = df['datetime'].apply(date_string_to_float)

		for df in valid_data_frames.values():
			df['datetime'] = df['datetime'].apply(date_string_to_float)

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	
	for name, df in train_data_frames.items():
		print(f'Started training model using {name}')
		train_lstm_model(name, df, 10, 32, 50)
	

	# Evalute models on test data
	for  name, df in valid_data_frames.items():
		print(f'Started evaluating model using {name}')
		test_lstm_model(name, df, 10)
