import pandas as pd

# Load CSV file
file_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/Raw Data/Program Menambahkan Province & Regency Automatic di File CSV/2022_after_gabung/output5.csv"  # Ganti dengan path file yang sesuai
df = pd.read_csv(file_path)

# Define the total hours in a year
total_hours = 8760000  # 8,760 * 1000

df_data = df.iloc[2:].apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns (if any)
df_data = df_data.dropna(axis=1, how='all')

# Calculate Capacity Factor (CF) for each column
cf_values = (df_data.sum() / total_hours) * 100

cf_row_data = ['Capacity Factor'] + cf_values.tolist()

# Adjust the length of CF row to match df.columns length
if len(cf_row_data) < len(df.columns):
    cf_row_data += [''] * (len(df.columns) - len(cf_row_data))
elif len(cf_row_data) > len(df.columns):
    cf_row_data = cf_row_data[:len(df.columns)]

cf_row = pd.DataFrame([cf_row_data], columns=df.columns)

# Append the CF row to the dataframe
df = pd.concat([df, cf_row], ignore_index=True)

output_file_path = "result_CF/output5_with_CF.csv"
df.to_csv(output_file_path, index=False)

print(f"File dengan Capacity Factor berhasil disimpan: {output_file_path}")
