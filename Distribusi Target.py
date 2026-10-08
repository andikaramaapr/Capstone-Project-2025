import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
# 2. Baca file CSV
df = pd.read_csv(file_path)

# Cek jumlah masing-masing kelas
print(df['Potability'].value_counts())

# Bisa juga lihat persentasenya
print(df['Potability'].value_counts(normalize=True) * 100)
