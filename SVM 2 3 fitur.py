import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Baca dataset
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
df = pd.read_csv(file_path)

# 2. Pilih hanya fitur 'ph', 'turbidity', dan 'solids' untuk training
X = df[['ph', 'Turbidity', 'Solids']]  # hanya fitur ini yang digunakan
y = df['Potability']

# 3. Imputasi nilai NaN dengan mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Split dataset jadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Latih model SVM (kernel RBF + gamma=scale)
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# 7. Evaluasi akurasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model SVM (StandardScaler, Fitur Terpilih): {accuracy * 100:.2f}%")

# 8. Prediksi satu sampel
sample = X_test[0]
predicted_class = model.predict([sample])[0]
print(f"Prediksi untuk sampel: {sample} adalah kelas {predicted_class}")
