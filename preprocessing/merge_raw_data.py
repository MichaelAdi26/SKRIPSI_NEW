import os
import pandas as pd

# # Path folder tempat file CSV/Excel berada
# folder_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/Raw Data/Program Menambahkan Province & Regency Automatic di File CSV/2023_before_gabung"

# # Ambil daftar file dalam folder
# file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.xlsx')]

# dataframes = []

# for idx, file in enumerate(file_list):
#     file_path = os.path.join(folder_path, file)
    
#     if file.endswith('.csv'):
#         df = pd.read_csv(file_path, header=None, low_memory=False, dtype=str)  # Paksa semua kolom menjadi string

#     # Pastikan format sesuai (minimal 4 baris)
#     if df.shape[0] < 4:
#         print(f"File {file} memiliki format yang tidak sesuai, dilewati!")
#         continue
    
#     # Jika ini adalah file pertama, ambil seluruh kolom
#     if idx == 0:
#         dataframes.append(df)
#     else:
#         # Untuk file kedua dan seterusnya, hanya ambil dari kolom kedua sampai terakhir
#         dataframes.append(df.iloc[:, 1:])

# # Gabungkan semua file secara horizontal (axis=1)
# merged_df = pd.concat(dataframes, axis=1)

# # Pastikan semua kolom memiliki nama unik (hindari duplikat)
# merged_df.columns = range(merged_df.shape[1])

# # Konversi semua kolom ke string untuk menghindari tipe data campuran
# merged_df = merged_df.astype(str).fillna("")

# # Path output untuk file Parquet
# path_output = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah"
# output_parquet = os.path.join(path_output, "2023_merged_before_geo_matching.parquet")

# # Simpan ke Parquet
# merged_df.to_parquet(output_parquet, engine="pyarrow", compression="snappy")

# print(f"Penggabungan selesai! File disimpan sebagai:\n- {output_parquet}")

df = pd.read_parquet("C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2023_merged_before_geo_matching.parquet")

print(df)