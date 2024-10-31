import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Memuat dataset
df = pd.read_csv('german_credit_data.csv')

# Memilih fitur numerik untuk normalisasi
features = df.select_dtypes(include=['float64', 'int64']).columns

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Normalisasi data
df[features] = scaler.fit_transform(df[features])

# Menyimpan data yang dinormalisasi ke file baru
df.to_csv('german_credit_data_normalized.csv', index=False)
print("Data yang dinormalisasi telah disimpan ke 'german_credit_data_normalized.csv'.")
