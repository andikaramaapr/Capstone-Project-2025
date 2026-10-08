import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Baca dataset
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (2).csv')
df = pd.read_csv(file_path)

# 2. Pilih hanya fitur dan label
X_raw = df[['ph', 'Solids', 'Turbidity']].values
y     = df['Potability'].values

# 3. Imputasi missing values (fit_transform pada X_raw)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_raw)

# 4. Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# (Opsional) Simpan scaler agar bisa reuse di ESP32
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

# 5. Split data: kita split **imputed** + **scaled** + **label** secara paralel
X_train_scaled, X_test_scaled, X_train_imp, X_test_imp, y_train, y_test = train_test_split(
    X_scaled, X_imputed, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Latih model SVM (linear kernel) pada data scaled
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# 8. Evaluasi akurasi
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Model SVM: {acc * 100:.2f}%")

# 9. Lakukan prediksi satu sampel (scaled)
sample = X_test_scaled[0]
predicted_class = model.predict([sample])[0]
print(f"Prediksi untuk sampel (scaled): {sample} adalah kelas {predicted_class}")

# 10. Ekstrak parameter model untuk header C
weights = model.coef_[0]   # shape (3,)
bias    = model.intercept_[0]

# 11. Generate C header
header = [
    "// Auto-generated SVM model header for TinyML on ESP32",
    "#ifndef SVM_MODEL_H",
    "#define SVM_MODEL_H",
    "",
    f"#define SVM_NUM_FEATURES {len(weights)}",
    "",
    "static const float svm_weights[SVM_NUM_FEATURES] = { " +
    ", ".join(f"{w:.6f}f" for w in weights) + " };",
    f"static const float svm_bias = {bias:.6f}f;",
    "",
    "static inline int svm_predict(const float features[SVM_NUM_FEATURES]) {",
    "    float sum = svm_bias;",
    "    for (int i = 0; i < SVM_NUM_FEATURES; ++i) {",
    "        sum += svm_weights[i] * features[i];",
    "    }",
    "    return sum >= 0.0f ? 1 : 0;",
    "}",
    "",
    "#endif  // SVM_MODEL_H",
]

with open("svm_model.h", "w") as f:
    f.write("\n".join(header))
print("[✓] Header file 'svm_model.h' berhasil dibuat.")
