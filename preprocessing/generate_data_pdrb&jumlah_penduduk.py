import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Data input Hardcode
jumlah_penduduk_awal = 10_640_010
jumlah_penduduk_akhir = 10_672_100
data_jumlah_penduduk_miskin_awal = 498_290
data_jumlah_penduduk_miskin_akhir = 494_930
data_pdrb_awal = 274_660
data_pdrb_akhir = 299_675


def generate_monthly_data(year=2022, initial_value=0, final_value=0):
    months_in_year = 12
    
    # indeks waktu dari Januari 2021 sampai Desember 2021 setiap bulan
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
date_range, penduduk_per_period = generate_monthly_data(2022, jumlah_penduduk_awal, jumlah_penduduk_akhir)
_, jumlah_penduduk_miskin_per_period = generate_monthly_data(2022, data_jumlah_penduduk_miskin_awal, data_jumlah_penduduk_miskin_akhir)
_, pdrb_per_period = generate_monthly_data(2022, data_pdrb_awal, data_pdrb_akhir)

data_per_period = pd.DataFrame({
    "Timestamp": date_range,
    "Jumlah_Penduduk": penduduk_per_period,
    "Jumlah_Penduduk_Miskin": jumlah_penduduk_miskin_per_period,
    "PDRB": pdrb_per_period
})

excel_filename = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/Data Demografi dan PDRB/Jakarta/Data_Jumlah_Penduduk_Jakarta_update_Per_Bulan_2022_ALLDATA.xlsx"
data_per_period.to_excel(excel_filename, index=False, engine='openpyxl')

print(data_per_period.head())

# Visualisasi
plt.figure(figsize=(12, 6))
plt.plot(data_per_period['Timestamp'], data_per_period['Jumlah_Penduduk'], label="Jumlah Penduduk")
plt.plot(data_per_period['Timestamp'], data_per_period['Jumlah_Penduduk_Miskin'], label="Jumlah Penduduk Miskin")
plt.plot(data_per_period['Timestamp'], data_per_period['PDRB'], label="PDRB")
plt.title("Interpolasi Data Penduduk per Bulan 2023 (Regresi Linear)")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.legend()
plt.grid()
plt.show()

print(f"Data berhasil disimpan sebagai {excel_filename}")
