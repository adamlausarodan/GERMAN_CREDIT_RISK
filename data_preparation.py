# Import library yang diperlukan
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Memuat dataset dari file CSV
df = pd.read_csv('german_credit_data.csv')

# Melihat sekilas data
print("Data sebelum One-Hot Encoding:")
print(df.head())

# Melihat tipe data dan kolom
print("\nInformasi dataset:")
print(df.info())

# Identifikasi kolom yang perlu di-encode (biasanya yang memiliki tipe 'object')
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nFitur kategorikal yang akan di-encode:", categorical_cols)

# Menggunakan pandas get_dummies untuk one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Menampilkan hasil one-hot encoding
print("\nData setelah One-Hot Encoding:")
print(df_encoded.head())

# Menyimpan data hasil encoding ke file baru
df_encoded.to_csv('german_credit_data_encoded.csv', index=False)
print("\nData hasil One-Hot Encoding telah disimpan ke 'german_credit_data_encoded.csv'.")
