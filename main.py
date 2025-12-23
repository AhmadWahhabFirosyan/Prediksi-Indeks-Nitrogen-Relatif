import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List

# Inisialisasi App
app = FastAPI(title="N-Prediction API")

# --- 1. LOAD MODEL & SCALERS ---
try:
    model = tf.keras.models.load_model('model/lstm_N_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    sc_dyn = joblib.load('model/scaler_dynamic.pkl')
    sc_stat = joblib.load('model/scaler_static.pkl')
    sc_y = joblib.load('model/scaler_target.pkl')
    print("Scalers loaded successfully.")
except Exception as e:
    print(f"Error loading scalers: {e}")

# --- 2. DEFINE INPUT SCHEMA ---
class PredictionInput(BaseModel):
    dynamic_data: List[List[float]] 
    static_data: List[float]

# --- 3. ORIGINAL JSON ENDPOINT ---
@app.post("/predict")
def predict_nitrogen(payload: PredictionInput):
    try:
        if len(payload.dynamic_data) != 24:
            raise HTTPException(status_code=400, detail="Dynamic data must be exactly 24 time steps.")
        
        X_dyn_raw = np.array(payload.dynamic_data)
        X_stat_raw = np.array(payload.static_data)

        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        urea_kg_ha = 275 + (y_pred_real * 62.5)
        urea_kg_ha = max(200, min(400, urea_kg_ha))

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

# --- 4. CSV PREDICTION ENDPOINT (BARU) ---
# DITEMPATKAN SEBELUM MAIN BLOCK
@app.post("/predict-csv")
async def predict_nitrogen_from_csv(
    file: UploadFile = File(...), 
    location_name: str = Form(...)
):
    try:
        # 1. Baca File CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Filter Berdasarkan Lokasi (NAME_3)
        df_loc = df[df['NAME_3'] == location_name].copy()
        
        if df_loc.empty:
            raise HTTPException(status_code=400, detail=f"Lokasi '{location_name}' tidak ditemukan di CSV.")

        # 3. Urutkan Berdasarkan Waktu
        df_loc['period_start'] = pd.to_datetime(df_loc['period_start'])
        df_loc = df_loc.sort_values('period_start')

        # 4. Ambil 24 Data Terakhir
        if len(df_loc) < 24:
            raise HTTPException(status_code=400, detail=f"Data untuk '{location_name}' kurang dari 24 baris. Hanya ada {len(df_loc)}.")
            
        df_window = df_loc.tail(24)
        
        # 5. Siapkan Data Dinamis
        dynamic_cols = ['NDVI', 'EVI', 'OSAVI', 'LST_mean', 'SoilMoisture', 'precip_sum']
        missing_cols = [c for c in dynamic_cols if c not in df_window.columns]
        if missing_cols:
             raise HTTPException(status_code=400, detail=f"Kolom dinamis hilang di CSV: {missing_cols}")

        X_dyn_raw = df_window[dynamic_cols].values 

        # 6. Siapkan Data Statis
        static_cols = ['pH', 'SOC', 'cec', 'clay', 'sand', 'elevation_m']
        missing_static = [c for c in static_cols if c not in df_window.columns]
        if missing_static:
             raise HTTPException(status_code=400, detail=f"Kolom statis hilang di CSV: {missing_static}")
             
        X_stat_raw = df_window[static_cols].iloc[-1].values

        # --- PREDIKSI ---
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        urea_kg_ha = 275 + (y_pred_real * 62.5)
        urea_kg_ha = max(200, min(400, urea_kg_ha))

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

        return {
            "status": "success",
            "meta": {
                "location": location_name,
                "data_points": 24,
                "last_date": str(df_window['period_start'].iloc[-1].date())
            },
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

# --- MAIN EXECUTION BLOCK (HARUS PALING BAWAH) ---
if __name__ == "__main__":
    import uvicorn
    # Jalankan di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)