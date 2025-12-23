import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import io
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List

# Inisialisasi App
app = FastAPI(title="Siladan N-Prediction API")

# --- 1. LOAD MODEL & SCALERS ---
try:
    model = tf.keras.models.load_model('model/lstm_N_model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    sc_dyn = joblib.load('model/scaler_dynamic.pkl')
    sc_stat = joblib.load('model/scaler_static.pkl')
    sc_y = joblib.load('model/scaler_target.pkl')
    print("✅ Scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading scalers: {e}")

# --- 2. DEFINE INPUT SCHEMA (Untuk JSON) ---
class PredictionInput(BaseModel):
    # Data Dinamis: 24 periode (per 15 hari)
    dynamic_data: List[List[float]] 
    # Data Statis: 6 fitur tanah
    static_data: List[float]

# --- 3. ENDPOINT JSON (ORIGINAL) ---
@app.post("/predict")
def predict_nitrogen(payload: PredictionInput):
    try:
        # A. VALIDASI
        if len(payload.dynamic_data) != 24:
            raise HTTPException(status_code=400, detail="Dynamic data must be exactly 24 time steps (24 periods).")
        
        # B. PREPROCESSING
        X_dyn_raw = np.array(payload.dynamic_data)
        X_stat_raw = np.array(payload.static_data)

        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        # C. PREDIKSI
        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        # D. REKOMENDASI
        return generate_recommendation(y_pred_real)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. ENDPOINT CSV (BARU & DIPERBAIKI) ---
@app.post("/predict-csv")
async def predict_nitrogen_from_csv(
    file: UploadFile = File(...), 
    location_name: str = Form(...)
):
    try:
        # 1. Baca File CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Filter Lokasi (Kolom NAME_3)
        if 'NAME_3' not in df.columns:
            raise HTTPException(status_code=400, detail="Kolom 'NAME_3' tidak ditemukan di CSV.")
            
        df_loc = df[df['NAME_3'] == location_name].copy()
        
        if df_loc.empty:
            raise HTTPException(status_code=404, detail=f"Lokasi '{location_name}' tidak ditemukan di dataset.")

        # 3. Urutkan Waktu
        if 'period_start' not in df_loc.columns:
             raise HTTPException(status_code=400, detail="Kolom 'period_start' wajib ada.")
             
        df_loc['period_start'] = pd.to_datetime(df_loc['period_start'])
        df_loc = df_loc.sort_values('period_start')

        # 4. Ambil 24 Data Terakhir
        if len(df_loc) < 24:
            raise HTTPException(status_code=400, detail=f"Data kurang. Lokasi '{location_name}' hanya punya {len(df_loc)} baris data (butuh min 24).")
            
        df_window = df_loc.tail(24)

        # --- VALIDASI TIME STEP (OPTIMISASI) ---
        # Cek apakah 24 data ini mencakup kurun waktu sekitar 1 tahun (360 hari)
        start_date = df_window['period_start'].iloc[0]
        end_date = df_window['period_start'].iloc[-1]
        days_diff = (end_date - start_date).days
        
        time_warning = None
        # Toleransi: idealnya ~345 hari (23 interval * 15 hari). 
        # Jika selisih > 400 hari, berarti ada data bolong.
        if days_diff > 400:
            time_warning = f"Data tidak kontinu! 24 periode mencakup {days_diff} hari (seharusnya ~345-360 hari). Hasil mungkin kurang akurat."
        elif days_diff < 300:
            time_warning = f"Rentang waktu terlalu pendek ({days_diff} hari). Pastikan data per 15 hari."

        # 5. Siapkan Fitur
        dynamic_cols = ['NDVI', 'EVI', 'OSAVI', 'LST_mean', 'SoilMoisture', 'precip_sum']
        static_cols = ['pH', 'SOC', 'cec', 'clay', 'sand', 'elevation_m']
        
        # Cek kelengkapan kolom
        missing = [c for c in dynamic_cols + static_cols if c not in df_window.columns]
        if missing:
             raise HTTPException(status_code=400, detail=f"Kolom berikut hilang di CSV: {missing}")

        X_dyn_raw = df_window[dynamic_cols].values 
        # Ambil data statis dari baris terakhir
        X_stat_raw = df_window[static_cols].iloc[-1].values 

        # 6. Scaling & Prediksi
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        # 7. Generate Output
        result = generate_recommendation(y_pred_real)
        
        # Tambahkan Meta Info
        result['meta'] = {
            "location": location_name,
            "data_period_start": str(start_date.date()),
            "data_period_end": str(end_date.date()),
            "total_days_covered": days_diff,
            "warning": time_warning
        }
        
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- HELPER FUNCTION ---
def generate_recommendation(n_rel_val):
    """Fungsi pembantu untuk menghitung rekomendasi pupuk"""
    # Rumus: 275 + (N_rel * 62.5)
    urea_kg_ha = 275 + (n_rel_val * 62.5)
    
    # Clip (Batasi nilai 200 - 400)
    urea_kg_ha = max(200, min(400, urea_kg_ha))

    # Kategori
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
            "n_rel_index": float(n_rel_val),
            "urea_kg_ha": float(urea_kg_ha),
            "kategori": kategori,
            "rekomendasi": rekomendasi_text
        }
    }

# --- MAIN EXECUTION BLOCK ---
# PERHATIAN: Ini harus selalu di paling bawah file
if __name__ == "__main__":
    import uvicorn
    # Jalankan di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)