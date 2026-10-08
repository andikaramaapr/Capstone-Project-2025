import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
# 2. Baca file CSV
df = pd.read_csv(file_path)

# Pisahkan berdasarkan kelas
df_layak = df[df['Potability'] == 1]
df_tidak_layak = df[df['Potability'] == 0]

# Undersample data tidak layak minum agar jumlahnya sama dengan data layak
df_tidak_layak_sampled = df_tidak_layak.sample(n=len(df_layak), random_state=42)

# Gabungkan kembali menjadi dataset seimbang
df_seimbang = pd.concat([df_layak, df_tidak_layak_sampled], axis=0)

# Acak ulang baris
df_seimbang = df_seimbang.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan ke file CSV baru
df_seimbang.to_csv("water_potability (3).csv", index=False)

# Tampilkan jumlah masing-masing kelas
jumlah_per_kelas = df_seimbang['Potability'].value_counts()
print("Jumlah data setelah disampling:")
print(jumlah_per_kelas)
