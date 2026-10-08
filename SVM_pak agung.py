import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Baca dataset
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
df = pd.read_csv(file_path)

# 2. Pisahkan fitur dan label
X = df.drop('Potability', axis=1)
y = df['Potability']

# 3. Imputasi nilai NaN dengan mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Split dataset jadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Tentukan parameter yang akan dicari (GridSearch)
param_grid = {
    'C': [0.1, 1, 10, 100],                  # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 1],      # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly']      # Jenis kernel
}

# 7. Inisialisasi model SVM
svm_model = SVC()

# 8. Gunakan GridSearchCV untuk mencari hyperparameter terbaik
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# 9. Fit GridSearchCV dengan data training
grid_search.fit(X_train, y_train)

# 10. Tampilkan parameter terbaik yang ditemukan
print(f"Best hyperparameters: {grid_search.best_params_}")

# 11. Ambil model terbaik yang ditemukan oleh GridSearchCV
best_model = grid_search.best_estimator_

# 12. Evaluasi model terbaik pada data testing
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model SVM setelah tuning hyperparameter: {accuracy * 100:.2f}%")

# 13. Prediksi untuk satu sampel dari data testing
sample = X_test[0]
predicted_class = best_model.predict([sample])[0]
print(f"Prediksi untuk sampel: {sample} adalah kelas {predicted_class}")
