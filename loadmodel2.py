from sklearn.tree import export_text
import joblib

# Load model
model = joblib.load("decision_tree_model.pkl")

# Ekspor model ke teks (pohon decision tree dalam bentuk if-else)
tree_rules = export_text(model, feature_names=["ph", "Turbidity", "Solids"])
print(tree_rules)
