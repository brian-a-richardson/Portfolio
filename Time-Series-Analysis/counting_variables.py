"""
Data Counting Script
Date: 7/8/2024
Author:  Brian Richardson
Class:  DATA 670
"""

import os

from functions import *

if __name__ == "__main__":
	# Specify the folder path
	folder_path = "Cleaned Data"
	try:
		# Load all csv files
		data_frames, number_of_df, ex_time = load_csv_files(folder_path, 1, False)

		# Print the number of files loaded
		print()
		print(f"'{number_of_df}' csv files were converted into data frames.")
		print(f"'{ex_time}' seconds has elapsed.")
		print()

	except FileNotFoundError as e:
		print(e)

	except ValueError as e:
		print(e)

	# specify the categories
	momentum = ['adx_100', 'adxr_100', 'apo_100_200', 'aroon_100_200_aroondown', 'aroon_100_200_aroonup', 'aroonosc_100', 'bop_0', 'cci_100', 'cmo_100', 'dx_100', 'macd_100_200_50_macd', 'macd_100_200_50_macdsignal',
				'macd_100_200_50_macdhist', 'macdfix_100_macd', 'macdfix_100_macdsignal', 'macdfix_100_macdhist', 'mfi_100', 'minus_di_100', 'minus_dm_100', 'mom_100', 'plus_di_100', 'plus_dm_100', 'ppo_100_200',
				'roc_100', 'rocp_100', 'rocr_100', 'rocr100_100', 'rsi_100', 'stoch_100_200_slowk', 'stoch_100_200_slowd', 'stochf_100_200_fastk', 'stochf_100_200_fastd', 'stochrsi_100_200_fastk', 'stochrsi_100_200_fastd',
				'trix_100', 'ultosc_100_200_300', 'willr_100']

	volume = ['ad_0', 'adosc_100_200', 'obv_0', 'volume']

	volitility = ['atr_100', 'natr_100', 'trange_0']

	overlap = ['bbands_100_upperband', 'bbands_100_middleband','bbands_100_lowerband', 'dema_100', 'ema_100', 'kama_100', 'ma_100', 'mama_0_00_mama', 'mama_0_00_fama', 'midpoint_100',
				'midprice_100', 'sar_0', 'sarext_0', 'sma_100', 't3_100', 'tema_100', 'trima_100', 'wma_100']

	cycle = columns_to_keep = ['ht_dcperiod_0', 'ht_dcphase_0', 'ht_dcphasor_0_00_inphase', 'ht_dcphasor_0_00_quadrature', 'ht_sine_0_00_sine', 'ht_sine_0_00_leadsine', 'ht_trendmode_0']

	# Get the column names
	first_key = next(iter(data_frames))
	columns = data_frames[first_key].columns

	# Set up the counters
	momentum_count = 0
	volume_count = 0
	volitility_count = 0
	overlap_count = 0
	cycle_count = 0

	# Loop and count
	for col in columns:
		if col in momentum:
			momentum_count += 1
		elif col in volume:
			volume_count += 1
		elif col in volitility:
			volitility_count += 1
		elif col in overlap:
			overlap_count += 1
		elif col in cycle:
			cycle_count += 1

	# Print the values
	print("The values are:")
	print(f"Momentum: {momentum_count}")
	print(f"Volume: {volume_count}")
	print(f"Volitility: {volitility_count}")
	print(f"Overlap: {overlap_count}")
	print(f"Cycle: {cycle_count}")