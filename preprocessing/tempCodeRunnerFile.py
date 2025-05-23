# Pastikan kolom nama benar (hapus spasi tambahan)
# df.columns = df.columns.str.strip()

# # Ambil data pelanggan listrik tahun 2021
# if "2021" in df.columns:
#     data_tahunan = df["2021"].values  # Ambil data sebagai array
# else:
#     print("Kolom '2021' tidak ditemukan dalam data.")
#     data_tahunan = None

# # Ambil nama provinsi
# if "PROVINSI" in df.columns:
#     provinsi = df["PROVINSI"].values  # Ambil nama provinsi
# else:
#     print("Kolom 'PROVINSI' tidak ditemukan dalam data.")
#     provinsi = None

# # Debugging: Menampilkan beberapa baris data awal
# print("Dataframe hasil pembacaan:\n", df.head(7))

# # Total jam dalam setahun
# total_jam = 24 * 365  # Jumlah jam dalam setahun (8760 jam)

# # Membuat DataFrame untuk hasil interpolasi (placeholder)
# result_df = pd.DataFrame()
