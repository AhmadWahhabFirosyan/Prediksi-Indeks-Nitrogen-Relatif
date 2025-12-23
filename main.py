import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Inisialisasi App
app = FastAPI(title="Siladan N-Prediction API")

# --- 1. LOAD MODEL & SCALERS ---
# Load model .h5
try:
    model = tf.keras.models.load_model('model/lstm_N_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load scalers .pkl
try:
    sc_dyn = joblib.load('model/scaler_dynamic.pkl')
    sc_stat = joblib.load('model/scaler_static.pkl')
    sc_y = joblib.load('model/scaler_target.pkl')
    print("Scalers loaded successfully.")
except Exception as e:
    print(f"Error loading scalers: {e}")

# --- 2. DEFINE INPUT SCHEMA ---
# Sesuai notebook, input dinamis butuh window size 24 jam
class PredictionInput(BaseModel):
    # Data Dinamis: List of 24 items. Tiap item berisi 6 fitur:
    # [NDVI, EVI, OSAVI, LST_mean, SoilMoisture, precip_sum]
    dynamic_data: List[List[float]] 
    
    # Data Statis: 1 item berisi 6 fitur:
    # [pH, SOC, cec, clay, sand, elevation_m]
    static_data: List[float]

# --- 3. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_nitrogen(payload: PredictionInput):
    try:
        # A. VALIDASI INPUT
        # Pastikan data dinamis ada 24 baris (window size)
        if len(payload.dynamic_data) != 24:
            raise HTTPException(status_code=400, detail="Dynamic data must be exactly 24 time steps.")
        
        # B. PREPROCESSING (Sama persis dengan Notebook Cell 12)
        # 1. Convert ke Numpy Array
        X_dyn_raw = np.array(payload.dynamic_data)  # Shape: (24, 6)
        X_stat_raw = np.array(payload.static_data)  # Shape: (6,)

        # 2. Reshape untuk Scaler
        # Scaler dinamis dilatih dengan data 2D, jadi kita flatten dulu jika perlu atau loop
        # Tapi sc_dyn.transform mengharapkan (n_samples, n_features).
        # Disini n_samples = 24, n_features = 6.
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)

        # 3. Reshape untuk Model LSTM (Batch Size, Window, Features)
        # Model butuh input (1, 24, 6)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        # 4. Scaling Static Data
        # Reshape ke (1, 6) karena scaler butuh 2D array
        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled # Shape (1, 6)

        # C. PREDIKSI (Sama dengan Notebook Cell 16)
        # Model hybrid menerima list input: [input_dinamis, input_statis]
        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        
        # Inverse transform untuk dapat nilai asli N_rel
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        # D. KONVERSI KE REKOMENDASI (Sama dengan Notebook Cell 17-19)
        # Rumus: 275 + (N_rel * 62.5)
        urea_kg_ha = 275 + (y_pred_real * 62.5)
        
        # Clip (Batasi nilai 200 - 400)
        urea_kg_ha = max(200, min(400, urea_kg_ha))

        # Tentukan Kategori
        kategori = ""
        rekomendasi_text = ""
        
        if urea_kg_ha <= 200:
            kategori = "Sangat Rendah"
            rekomendasi_text = "≥ 350 kg/ha"
        elif urea_kg_ha <= 250:
            kategori = "Rendah"
            rekomendasi_text = "200–250 kg/ha"
        elif urea_kg_ha <= 300:
            kategori = "Sedang"
            rekomendasi_text = "250–300 kg/ha (standar)"
        elif urea_kg_ha <= 350:
            kategori = "Tinggi"
            rekomendasi_text = "300–350 kg/ha"
        else:
            kategori = "Sangat Tinggi"
            rekomendasi_text = "≤ 300 kg/ha (kurangi dosis)"

        # E. RETURN JSON
        return {
            "status": "success",
            "prediction": {
                "n_rel_index": float(y_pred_real),
                "urea_kg_ha": float(urea_kg_ha),
                "kategori": kategori,
                "rekomendasi": rekomendasi_text
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Agar bisa dijalankan langsung dengan `python main.py`
if __name__ == "__main__":
    import uvicorn
    # Jalankan di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)