import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
# 2. Baca file CSV
df = pd.read_csv(file_path)
df.fillna(df.mean(), inplace=True)
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
