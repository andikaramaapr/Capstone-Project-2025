import joblib
from sklearn.tree import export_text

model = joblib.load("decision_tree_model.pkl")  # Ganti dengan nama file yang sesuai
print("Model berhasil dimuat!")

feature_names = ['ph', 'Turbidity', 'Solids']  # Sesuaikan dengan fitur yang kamu pakai saat training
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)
