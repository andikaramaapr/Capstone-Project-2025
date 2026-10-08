import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
file_path_X = r'D:\Semester 6\Capstone\Andika\Dataset\fitur_cleaned.csv'
file_path_y = r'D:\Semester 6\Capstone\Andika\Dataset\kelas_cleaned.csv'

X = pd.read_csv(file_path_X)[['ph', 'Turbidity', 'Solids']]
y = pd.read_csv(file_path_y).values.ravel()  # pastikan dalam 1D array

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kategori (anggap class-nya 3 kelas, bisa disesuaikan)
y_encoded = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Akurasi model MLP: {acc * 100:.2f}%")