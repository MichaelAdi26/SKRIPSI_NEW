import pandas as pd

# Path ke file CSV
file_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/result_CF/output4_with_CF.csv"

# Load file CSV dengan multi-header (Provinsi & Regency)
df = pd.read_csv(file_path, header=[0, 1])

# Gabungkan MultiIndex header menjadi satu string "Provinsi_Regency"
df.columns = ['_'.join(col).strip() for col in df.columns]

# Cek beberapa baris pertama
print("Preview DataFrame:")
print(df.head())

# Pastikan baris pertama setelah header adalah numerik (bukan tanggal)
if not df.iloc[0].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
    print("\n⚠️ Perhatian: Baris pertama tampaknya bukan data numerik!")
    print(df.iloc[0])  # Cek baris pertama
    df = df.iloc[1:]   # Hapus baris pertama jika perlu

# Ubah tipe data kolom yang seharusnya numerik
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Misal ingin mencari kolom dengan nilai maksimum
max_cf_column = df.iloc[:, 1:].max().idxmax()  # Kolom dengan nilai max tertinggi
max_cf_value = df[max_cf_column].max()  # Nilai maksimum

# Ambil koordinat atau informasi terkait berdasarkan kolom maksimum
max_coordinates = df.iloc[0, df.columns.get_loc(max_cf_column)]

print(f"\nKolom dengan nilai max tertinggi: {max_cf_column}")
print(f"Nilai maksimum: {max_cf_value}")
print(f"Koordinat terkait: {max_coordinates}")
