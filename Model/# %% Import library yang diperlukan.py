import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

file_path_X = r'D:\Semester 6\Capstone\Andika\Dataset\fitur_cleaned.csv'
file_path_y = r'D:\Semester 6\Capstone\Andika\Dataset\kelas_cleaned.csv'

X = pd.read_csv(file_path_X)
y = pd.read_csv(file_path_y)

X = X[['ph', 'Turbidity', 'Solids']]  # hanya fitur ini yang digunakan

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Default C=1.0 dan gamma='scale'
model = SVC(kernel='linear', C=0.0001) 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred)
print(f"Akurasi model pada data uji: {accuracy_test * 100:.2f}%")
