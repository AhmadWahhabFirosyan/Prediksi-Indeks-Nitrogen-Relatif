import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import traceback
import sys
from pydantic import BaseModel
from typing import List
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi App
app = FastAPI(title="N-Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua akses
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD MODEL & SCALERS ---
try:
    # compile=False agar tidak error metric mse
    model = tf.keras.models.load_model('model/lstm_model.h5', compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    sc_dyn = joblib.load('model/scaler_dynamic.pkl')
    sc_stat = joblib.load('model/scaler_static.pkl')
    sc_y = joblib.load('model/scaler_target.pkl')
    print("✅ Scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading scalers: {e}")
    sc_dyn = sc_stat = sc_y = None

# --- 2. DEFINE INPUT SCHEMA ---
class PredictionInput(BaseModel):
    dynamic_data: List[List[float]] 
    static_data: List[float]

# --- 3. HELPER FUNCTION (YANG DIPERBAIKI) ---
def generate_recommendation(n_rel_val):
    # 1. Hitung Status Skor (Angka asli status tanah)
    n_status_score = 275 + (n_rel_val * 62.5) 
    
    # Inisialisasi variabel agar aman
    final_dose = 0
    kategori = ""
    rekomendasi_text = ""

    # 2. Logika Penentuan Dosis (Konsisten menggunakan 'rekomendasi_text' & 'final_dose')
    if n_status_score <= 200:
        kategori = "Sangat Rendah"
        final_dose = 350
        rekomendasi_text = "Tambahkan ≥ 350 kg/ha"
        
    elif n_status_score <= 250:
        kategori = "Rendah"
        final_dose = 300
        rekomendasi_text = "Tambahkan 300-350 kg/ha" # Ditinggikan agar sesuai kebutuhan
        
    elif n_status_score <= 300:
        kategori = "Sedang"
        final_dose = 275
        rekomendasi_text = "Kondisi ideal. Pertahankan dosis standar (250–300 kg/ha)"
        
    elif n_status_score <= 350:
        kategori = "Tinggi"
        final_dose = 225  # Turunkan dosis
        rekomendasi_text = "Tanah subur. Kurangi dosis (Cukup 200-250 kg/ha)"
        
    else:
        kategori = "Sangat Tinggi"
        final_dose = 150
        rekomendasi_text = "Sangat subur! Hemat pupuk (≤ 200 kg/ha)"

    # 3. Return Hasil
    return {
        "status": "success",
        "prediction": {
            "n_rel_index": float(n_rel_val),
            "urea_kg_ha": float(final_dose),       # Mengirim angka Dosis Rekomendasi (bukan skor status)
            "status_score": float(n_status_score), # Info tambahan: Skor Status Asli
            "kategori": kategori,
            "rekomendasi": rekomendasi_text        # Variabel ini sekarang konsisten
        }
    }

# --- 4. ENDPOINT JSON (Manual Input) ---
@app.post("/predict")
def predict_nitrogen(payload: PredictionInput):
    if not model or not sc_dyn:
        raise HTTPException(status_code=503, detail="Model/Scalers not initialized.")

    try:
        if len(payload.dynamic_data) != 24:
            raise HTTPException(status_code=400, detail="Dynamic data must be exactly 24 time steps.")
        
        X_dyn_raw = np.array(payload.dynamic_data)
        X_stat_raw = np.array(payload.static_data)

        # Preprocessing
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        # Prediksi
        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        return generate_recommendation(y_pred_real)

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. ENDPOINT CSV (OPTIMIZED) ---
@app.post("/predict-csv")
def predict_nitrogen_from_csv(
    file: UploadFile = File(...), 
    location_name: str = Form(...)
):
    print(f"DEBUG: Menerima request untuk lokasi '{location_name}' dengan file '{file.filename}'")

    if not model or not sc_dyn:
        raise HTTPException(status_code=503, detail="Model/Scalers not initialized.")

    # 1. Validasi Tipe File
    if not file.filename.lower().endswith('.csv'):
        print(f"DEBUG: File ditolak karena ekstensi bukan .csv ({file.filename})")
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    try:
        # 2. Baca File
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
             print(f"DEBUG: Gagal membaca CSV. Error: {e}")
             raise HTTPException(status_code=400, detail="Gagal membaca file CSV. Pastikan format benar.")

        # 3. Filter Lokasi
        if 'NAME_3' not in df.columns:
            print("DEBUG: Kolom 'NAME_3' tidak ditemukan di CSV.")
            raise HTTPException(status_code=400, detail="Kolom 'NAME_3' tidak ditemukan.")
            
        df_loc = df[df['NAME_3'] == location_name].copy()
        
        if df_loc.empty:
            print(f"DEBUG: Lokasi '{location_name}' tidak ada di data.")
            available = df['NAME_3'].unique()[:5] 
            raise HTTPException(status_code=404, detail=f"Lokasi '{location_name}' tidak ditemukan. Tersedia: {available}...")

        # 4. Processing Waktu
        if 'period_start' not in df_loc.columns:
             raise HTTPException(status_code=400, detail="Kolom 'period_start' wajib ada.")
             
        df_loc['period_start'] = pd.to_datetime(df_loc['period_start'])
        df_loc = df_loc.sort_values('period_start')
        df_loc.set_index('period_start', inplace=True)

        # --- DEBUG GAP CHECK ---
        time_diff = df_loc.index.to_series().diff().dt.days
        max_gap = time_diff.max()
        print(f"DEBUG: Max gap data untuk {location_name} adalah {max_gap} hari.")
        
        if max_gap > 60:
             print(f"WARNING: Data {location_name} memiliki gap besar ({max_gap} hari). Hasil interpolasi mungkin bias.")

        # --- AUTO-REPAIR ---
        numeric_cols = df_loc.select_dtypes(include=[np.number]).columns
        
        df_resampled = df_loc[numeric_cols].resample('15D').mean().interpolate(method='linear')
        df_resampled = df_resampled.bfill(limit=3).ffill(limit=3)

        if len(df_resampled) < 24:
             print(f"DEBUG: Data kurang. Hasil resample cuma {len(df_resampled)} baris.")
             raise HTTPException(status_code=400, detail=f"Data kurang. Hanya tersedia {len(df_resampled)} periode 15-harian.")
             
        df_window = df_resampled.tail(24).reset_index()

        # 5. Validasi Kolom Fitur
        dynamic_cols = ['NDVI', 'EVI', 'OSAVI', 'LST_mean', 'SoilMoisture', 'precip_sum']
        static_cols = ['pH', 'SOC', 'cec', 'clay', 'sand', 'elevation_m']
        
        missing = [c for c in dynamic_cols + static_cols if c not in df_window.columns]
        if missing:
             print(f"DEBUG: Kolom hilang: {missing}")
             raise HTTPException(status_code=400, detail=f"Kolom fitur hilang: {missing}")

        # 6. Prediksi
        X_dyn_raw = df_window[dynamic_cols].values 
        X_stat_raw = df_window[static_cols].iloc[-1].values

        # Scaling
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        # Inference
        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        # Generate Recommendation
        result = generate_recommendation(y_pred_real)
        
        result['meta'] = {
            "location": location_name,
            "period_start": str(df_window['period_start'].iloc[0].date()),
            "period_end": str(df_window['period_start'].iloc[-1].date()),
            "debug_max_gap_days": float(max_gap) if not pd.isna(max_gap) else 0
        }
        
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        print(f"DEBUG: Unhandled Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)