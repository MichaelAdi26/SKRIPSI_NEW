import pandas as pd

# function untuk melakukan extraction  data-data ke file baru yang sudah di preprocessing:
def filter_large_csv(input_file, output_file, columns_to_keep):
    """
    Filter specific columns from a CSV file and save to a new file.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the filtered CSV file.
    :param columns_to_keep: List of column names to retain in the output.
    """
    # Read only the specified columns
    df = pd.read_csv(input_file, usecols=columns_to_keep)
    
    # Save the filtered data to the new file
    df.to_csv(output_file, index=False)

# Path to the input CSV file
input_file = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2023.csv"  # Ganti dengan path file 
# input_file = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2021_alldata_update.csv"
# input_file = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2022_alldata_update.csv"
# Columns to keep utk dataset tahun 2021 dan 2022 dimana terdapat regency/ kota:
# columns_to_keep = ['Date', 'Geo', 'Province', 'Regency', 'Temperature', 'Output']

# Columns to keep utk dataset tahun 203 dimana tdk ada regency/ kota:
columns_to_keep = ['Date', 'Geo', 'Province', 'Temperature', 'Output']

# Define the output file name --> nama file yang baru
output_file = "2023_All.csv"

filter_large_csv(input_file, output_file, columns_to_keep)

print(f"Filtered data has been saved to {output_file}")


# melakukan merge data dan create file baru yang berisi data terkait: demand output, temperature, penduduk, pelanggan listrik, pdrb, dan data listrik pln:
