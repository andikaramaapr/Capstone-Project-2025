import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib

# 1. Cari file CSV di direktori yang sama dengan file Python
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (1).csv')

# 2. Baca file CSV
df = pd.read_csv(file_path)

print("Data awal:")
print(df.head())

# 3. Ambil hanya kolom: ph, Turbidity, Solids, dan Potability
df = df[['ph', 'Turbidity', 'Solids', 'Potability']]

# 4. Isi missing value dengan nilai rata-rata
imputer = SimpleImputer(strategy='mean')
df[['ph', 'Turbidity', 'Solids']] = imputer.fit_transform(df[['ph', 'Turbidity', 'Solids']])

# 5. Pisahkan fitur dan label
X = df[['ph', 'Turbidity', 'Solids']]
y = df['Potability']

# 6. Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler disimpan ke file: scaler.pkl")

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Buat dan latih model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# 9. Evaluasi
y_pred = model_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Decision Tree: {accuracy:.4f}")

# 10. Simpan model
model_filename = "decision_tree_model.pkl"
joblib.dump(model_dt, model_filename)
print(f"Model Decision Tree disimpan ke file: {model_filename}")

