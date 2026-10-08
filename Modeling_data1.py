import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

# 7. Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model: Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, model_rf.predict(X_test))

# 9. Model: SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)
acc_svm = accuracy_score(y_test, model_svm.predict(X_test))

# 10. Model: KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
acc_knn = accuracy_score(y_test, model_knn.predict(X_test))

# 11. Cetak akurasi tiap model
print("\nAkurasi Model:")
print(f"Random Forest : {acc_rf:.4f}")
print(f"SVM           : {acc_svm:.4f}")
print(f"KNN           : {acc_knn:.4f}")

# 12. Simpan model terbaik
models = {
    "Random Forest": (model_rf, acc_rf),
    "SVM": (model_svm, acc_svm),
    "KNN": (model_knn, acc_knn)
}

best_model_name = max(models, key=lambda name: models[name][1])
best_model, best_accuracy = models[best_model_name]

print(f"Model terbaik: {best_model_name} dengan akurasi {best_accuracy:.4f}")

# 13. Simpan model
model_filename = f"{best_model_name.lower().replace(' ', '_')}_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Model disimpan ke file: {model_filename}")
