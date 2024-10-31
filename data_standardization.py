# Import library yang diperlukan
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Jika Anda juga melakukan standarisasi

# Memuat dataset dari file CSV
df = pd.read_csv('german_credit_data.csv')

# Memilih fitur numerik untuk standarisasi
features = df.select_dtypes(include=['float64', 'int64']).columns

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Standarisasi data
df[features] = scaler.fit_transform(df[features])

# Menyimpan data yang distandarisasi ke file baru
df.to_csv('german_credit_data_standardized.csv', index=False)
print("Data yang distandarisasi telah disimpan ke 'german_credit_data_standardized.csv'.")
