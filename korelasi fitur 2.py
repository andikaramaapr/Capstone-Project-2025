import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
# 2. Baca file CSV
df = pd.read_csv(file_path)

# Hitung korelasi
correlation = df.corr()

# Visualisasi korelasi
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(correlation, cmap='coolwarm')
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
fig.colorbar(cax)
plt.title("Correlation Matrix", pad=20)
plt.show()
