import numpy as np
import joblib
import pandas as pd

# 1. Load model dan scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Input baru dari sensor [ph, turbidity, solids]
# Ganti angka berikut dengan data yang ingin kamu uji
input_data = pd.DataFrame([[8.1, 3.5, 13699]], columns=['ph', 'Turbidity', 'Solids'])

# 3. Scaling input (agar sesuai model)
input_scaled = scaler.transform(input_data)

# 4. Prediksi hasil
prediction = model.predict(input_scaled)

# 5. Interpretasi hasil
if prediction[0] == 1:
    print("Prediksi: Layak Minum")
else:
    print("Prediksi: Tidak Layak")
