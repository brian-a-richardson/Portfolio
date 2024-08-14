import pandas as pd
import os

def delete_csv_files(folder_path, number_of_rows):
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

	for file in csv_files:
		file_path = os.path.join(folder_path, file)
		df = pd.read_csv(file_path)
		if len(df) <= number_of_rows:
			os.remove(file_path)
			print(f'{file} has been removed.')
		else:
			print(f'{file} has been kept.')

if __name__ == "__main__":
	folder_path = "Raw Data"
	delete_csv_files(folder_path, 1000)