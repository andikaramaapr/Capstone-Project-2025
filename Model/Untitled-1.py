# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# %%
# 1. Load dataset
file_path_X = r'D:\Semester 6\Capstone\Andika\Dataset\fitur_cleaned.csv'
file_path_y = r'D:\Semester 6\Capstone\Andika\Dataset\kelas_cleaned.csv'

X = pd.read_csv(file_path_X)[['ph', 'Turbidity', 'Solids']]
y = pd.read_csv(file_path_y)

# %%
# 2. Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# 3. SMOTE untuk balancing kelas
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# %%
# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# %%
# 5. Training model SVM dengan C=1000 dan gamma=1
model = SVC(kernel='rbf', C=1000, gamma=1)
model.fit(X_train, y_train)

# %%
# 6. Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi akhir (SVM + SMOTE, C=1000, gamma=1): {accuracy * 100:.2f}%")
