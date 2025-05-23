import pandas as pd

# Load the CSV file
file_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/Raw Data/Program Menambahkan Province & Regency Automatic di File CSV/2022_after_gabung/output3.csv"  # Ganti dengan path file yang sesuai
df = pd.read_csv(file_path)

# Define the total hours in a year
total_hours = 8760000  # 8,760 * 1000

# Skip the first two header rows (Provinsi, Regency) dan proses hanya data numerik
df_data = df.iloc[2:].apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns (if any)
df_data = df_data.dropna(axis=1, how='all')

# Calculate Capacity Factor (CF) for each column
cf_values = (df_data.sum() / total_hours) * 100

# Ensure CF row matches exactly the number of columns in df
cf_row_data = ['Capacity Factor'] + cf_values.tolist()

# Adjust the length of CF row to match df.columns length
if len(cf_row_data) < len(df.columns):
    cf_row_data += [''] * (len(df.columns) - len(cf_row_data))
elif len(cf_row_data) > len(df.columns):
    cf_row_data = cf_row_data[:len(df.columns)]

# Convert to DataFrame
cf_row = pd.DataFrame([cf_row_data], columns=df.columns)

# Append the CF row to the dataframe
df = pd.concat([df, cf_row], ignore_index=True)

# Find the maximum CF value and its location
max_cf_value = cf_values.max()
max_cf_column = cf_values.idxmax()

# Find the corresponding Regency and coordinates
max_regency = df.iloc[1][max_cf_column]  # Assuming Regency is in the second row
max_coordinates = df.iloc[0][max_cf_column]  # Assuming coordinates are in the first row

# Create a new row for max CF information
max_cf_row = pd.DataFrame([["Max CF", max_regency, max_coordinates, max_cf_value] + ["" for _ in range(len(df.columns) - 4)]], columns=df.columns)

# Append the max CF row to the dataframe
df = pd.concat([df, max_cf_row], ignore_index=True)

# Save the modified file
output_file_path = "result_CF/output3_with_MaxCF.csv"
df.to_csv(output_file_path, index=False)

print(f"File dengan Capacity Factor berhasil disimpan: {output_file_path}")
