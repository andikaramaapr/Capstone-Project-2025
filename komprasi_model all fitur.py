# 1. Import library
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
from sklearn.tree import DecisionTreeClassifier
import joblib

# 2. Cari file CSV di direktori yang sama dengan file Python
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')

# 3. Baca file CSV
df = pd.read_csv(file_path)

print("Data awal:")
print(df.head())

# 4. Tangani missing value untuk semua kolom kecuali Potability
features = df.drop('Potability', axis=1)
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# 5. Ambil label Potability
y = df['Potability']

# 6. Normalisasi semua fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_imputed)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# 7. Split data: 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Inisialisasi dan latih semua model
# Model 1: Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

# Model 2: SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)
pred_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm)

# Model 3: KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)

# Model 4: Decision Tree
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, pred_dt)

# 9. Tampilkan hasil akurasi
print("\nAkurasi Model (Semua Fitur):")
print(f"Random Forest : {acc_rf * 100:.2f}%")
print(f"SVM           : {acc_svm * 100:.2f}%")
print(f"KNN           : {acc_knn * 100:.2f}%")
print(f"Decision Tree : {acc_dt * 100:.2f}%")

# 10. Tentukan model terbaik
models = {
    "Random Forest": (model_rf, acc_rf),
    "SVM": (model_svm, acc_svm),
    "KNN": (model_knn, acc_knn),
    "Decision Tree": (model_dt, acc_dt)
}

best_model_name = max(models, key=lambda name: models[name][1])
best_model, best_accuracy = models[best_model_name]

print(f"Model terbaik: {best_model_name} dengan akurasi {best_accuracy * 100:.2f}%")
