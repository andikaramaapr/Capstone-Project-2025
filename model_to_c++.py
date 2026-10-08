import joblib
from sklearn.tree import _tree

# Load model dan scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Fitur input (harus sesuai training)
feature_names = ["ph", "Turbidity", "Solids"]
means = scaler.mean_
scales = scaler.scale_

# Nama file output
output_file = "decision_tree_model.cpp"

# Fungsi konversi tree ke if-else C++
def tree_to_code_cpp(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    lines = []

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            lines.append(f"{indent}if ({name} <= {threshold:.6f}) {{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            value = tree_.value[node]
            class_id = int(value.argmax())
            lines.append(f"{indent}potability = {class_id};")

    # Header + Scaling
    lines.append("// === Decision Tree Predict Function ===")
    lines.append("int predict_potability(float ph_raw, float turbidity_raw, float solids_raw) {")
    lines.append("    int potability = -1;")

    # Tambahkan normalisasi
    lines.append(f"    float ph = (ph_raw - {means[0]:.6f}) / {scales[0]:.6f};")
    lines.append(f"    float Turbidity = (turbidity_raw - {means[1]:.6f}) / {scales[1]:.6f};")
    lines.append(f"    float Solids = (solids_raw - {means[2]:.6f}) / {scales[2]:.6f};")

    recurse(0, 1)
    lines.append("    return potability;")
    lines.append("}")

    return "\n".join(lines)

# Generate dan simpan file C++
cpp_code = tree_to_code_cpp(model, feature_names)
with open(output_file, "w") as f:
    f.write(cpp_code)

print(f"[✓] Model berhasil diekspor ke file: {output_file}")
