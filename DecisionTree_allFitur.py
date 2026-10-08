import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================================
# Fungsi-fungsi ID3
# ================================

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def information_gain(X, y, feature_index, threshold):
    parent_entropy = entropy(y)
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0
    child_entropy = (len(y_left) / len(y)) * entropy(y_left) + (len(y_right) / len(y)) * entropy(y_right)
    return parent_entropy - child_entropy

def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]
        feature, threshold = best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
        self.tree = {
            'feature': feature,
            'threshold': threshold,
            'left': self.fit(X_left, y_left, depth + 1),
            'right': self.fit(X_right, y_right, depth + 1)
        }
        return self.tree

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = tree['feature']
        threshold = tree['threshold']
        if x[feature] <= threshold:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

# ================================
# Load dan Preprocessing Dataset Water Potability
# ================================

# Ubah path ini sesuai lokasi file kamu
# 1. Cari file CSV di direktori yang sama dengan file Python
file_path = os.path.join(os.path.dirname(__file__), 'water_potability (3).csv')

# 2. Baca file CSV
df = pd.read_csv(file_path)
# Pisahkan fitur dan label (menggunakan semua kolom kecuali 'Potability' sebagai fitur)
X = df.drop('Potability', axis=1).values
y = df['Potability'].values

# Isi missing value dengan rata-rata
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ================================
# Training Model ID3
# ================================

model = DecisionTreeID3(max_depth=5)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model ID3 (Semua Fitur): {accuracy * 100:.2f}%")

# Prediksi satu data
sample_data = np.array([X_test[0]])
predicted_class = model.predict(sample_data)[0]
print(f'Prediksi untuk sampel: {sample_data[0]} adalah kelas {predicted_class}')
