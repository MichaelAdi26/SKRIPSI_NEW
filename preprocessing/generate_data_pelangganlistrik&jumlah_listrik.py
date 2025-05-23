import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Data input Hardcode
jumlah_pelanggan_awal = 1_630_528
jumlah_pelanggan_akhir = 1_701_512
listrik_terjual_awal = 5_470_511_000
listrik_terjual_akhir = 6_354_530_000
daya_terpasang_awal = 4_213_000
daya_terpasang_akhir = 4_537_000
produksi_listrik_awal = 5_695_849_000
produksi_listrik_akhir = 6_646_390_000

def generate_monthly_data(year=2023, initial_value=0, final_value=0):
    months_in_year = 12
    
    # Buat indeks waktu dari Januari 2021 sampai Desember 2021 setiap bulan
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')
    
    # variabel waktu sebagai fitur untuk regresi linear
    X = np.arange(months_in_year).reshape(-1, 1)
    
    # target nilai dari awal ke akhir (interpolasi linier)
    y = np.linspace(initial_value, final_value, months_in_year)
    
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    
    return date_range, trend

# Generate interpolasi untuk setiap variabel
date_range, pelanggan_per_period = generate_monthly_data(2023, jumlah_pelanggan_awal, jumlah_pelanggan_akhir)
_, listrik_terjual_per_period = generate_monthly_data(2023, listrik_terjual_awal, listrik_terjual_akhir)
_, daya_terpasang_per_period = generate_monthly_data(2023, daya_terpasang_awal, daya_terpasang_akhir)
_, produksi_listrik_per_period = generate_monthly_data(2023, produksi_listrik_awal, produksi_listrik_akhir)

data_per_period = pd.DataFrame({
    "Timestamp": date_range,
    "Jumlah_Pelanggan": pelanggan_per_period,
    "Listrik_Terjual": listrik_terjual_per_period,
    "Daya_Terpasang": daya_terpasang_per_period,
    "Produksi_Listrik": produksi_listrik_per_period
})

excel_filename = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/Data Pelanggan Listrik, Daya Terpasang, Listrik Terjual, dan Produksi Listrik/Bali/Data_Listrik_per_bulan_Bali_2023_ALLDATA.xlsx"
data_per_period.to_excel(excel_filename, index=False, engine='openpyxl')

print(data_per_period.head())

# Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(data_per_period['Timestamp'], data_per_period['Jumlah_Pelanggan'], label="Jumlah Pelanggan")
plt.plot(data_per_period['Timestamp'], data_per_period['Listrik_Terjual'], label="Listrik Terjual")
plt.plot(data_per_period['Timestamp'], data_per_period['Daya_Terpasang'], label="Daya Terpasang")
plt.plot(data_per_period['Timestamp'], data_per_period['Produksi_Listrik'], label="Produksi Listrik")
plt.title("Interpolasi Data Listrik per Bulan 2021 (Regresi Linear)")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.legend()
plt.grid()
plt.show()

print(f"Data berhasil disimpan sebagai {excel_filename}")
