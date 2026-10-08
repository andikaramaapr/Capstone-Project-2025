import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
from imblearn.over_sampling import SMOTE

# 1. Cari file CSV di direktori yang sama dengan file Python
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')

# 2. Baca file CSV
df = pd.read_csv(file_path)

print("Data awal:")
print(df.head())

# 3. Ambil hanya kolom: ph, Hardness, Turbidity, Solids, dan Potability
df = df[['ph', 'Hardness', 'Turbidity', 'Solids', 'Potability']]

# 4. Isi missing value dengan nilai rata-rata
imputer = SimpleImputer(strategy='mean')
df[['ph','Hardness', 'Turbidity', 'Solids']] = imputer.fit_transform(df[['ph','Hardness', 'Turbidity', 'Solids']])

# 5. Pisahkan fitur dan label
X = df[['ph','Hardness', 'Turbidity', 'Solids']]
y = df['Potability']

# 6. Scaling fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler disimpan ke file: scaler.pkl")

# 7. Split data (penting: split dulu sebelum SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Gabungkan X_test dan y_test menjadi satu DataFrame
test_data_df = pd.DataFrame(X_test, columns=['ph','Hardness', 'Turbidity', 'Solids'])
test_data_df['Potability'] = y_test.values  # Menambahkan kolom 'Potability'

# Simpan test set (dengan Potability) ke CSV
test_data_df.to_csv("test_data_with_potability.csv", index=False)
print("Data test (20%) disimpan ke file: test_data_with_potability.csv")

# 8. Terapkan SMOTE pada data training
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Jumlah data sebelum SMOTE: {len(y_train)}")
print(f"Jumlah data setelah SMOTE : {len(y_train_resampled)}")

# 9. Buat dan latih model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_resampled, y_train_resampled)

# 10. Evaluasi
y_pred = model_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Decision Tree: {accuracy:.4f}")

# 10. Simpan model
model_filename = "decision_tree_model.pkl"
joblib.dump(model_dt, model_filename)
print(f"Model Decision Tree disimpan ke file: {model_filename}")
