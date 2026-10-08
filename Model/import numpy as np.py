import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Your existing code
file_path_X = r'D:\Semester 6\Capstone\Andika\Dataset\fitur_cleaned.csv'
file_path_y = r'D:\Semester 6\Capstone\Andika\Dataset\kelas_cleaned.csv'

X = pd.read_csv(file_path_X)
y = pd.read_csv(file_path_y)

X = X[['ph', 'Turbidity', 'Solids']]  # hanya fitur ini yang digunakan

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Try different C values to reduce support vectors
models = []

# Try much smaller C values (promotes fewer support vectors)
for c_value in [0.001, 0.01, 0.1, 1.0]:
    model = SVC(kernel='linear', C=c_value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    sv_count = model.n_support_.sum()
    
    print(f"C={c_value}, Accuracy: {accuracy * 100:.2f}%, Support Vectors: {sv_count}")
    models.append((model, accuracy, sv_count, c_value))

# Sort by number of support vectors (ascending)
models.sort(key=lambda x: x[2])

# Choose the model with fewest support vectors that still has good accuracy
best_model, best_accuracy, sv_count, c_value = models[0]
print("\nSelected model:")
print(f"C={c_value}, Accuracy: {best_accuracy * 100:.2f}%, Support Vectors: {sv_count}")
print("\nDetailed report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the mean and std from scaler for use in Arduino
scaler_means = scaler.mean_
scaler_scales = scaler.scale_

print("\nScaling parameters for Arduino:")
print(f"Mean values: {scaler_means}")
print(f"Scale values: {scaler_scales}")

# Now export to Arduino
try:
    from eloquentarduino.ml.classifier import SVMClassifier
    
    # Use reduced precision to save memory
    eml_classifier = SVMClassifier(best_model, precision=3)
    
    # Generate Arduino code
    arduino_code = eml_classifier.to_arduino(instance_name="classifier")
    
    # Save to file
    with open("SVMClassifier.h", "w") as file:
        file.write(arduino_code)
    
    print(f"\nModel saved to SVMClassifier.h")
    print(f"Code size: {len(arduino_code)} characters")
    print(f"Support vectors: {len(eml_classifier.sv)}")
    
except ImportError:
    print("\nTo export to Arduino, install the eloquent-arduino package:")
    print("pip install eloquent-arduino")
    print("\nThen run this script again.")