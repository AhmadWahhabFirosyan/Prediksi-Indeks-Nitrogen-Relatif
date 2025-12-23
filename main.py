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
app = FastAPI(title="N-Prediction API")

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

# --- 2. DEFINE INPUT SCHEMA (Untuk JSON Manual) ---
class PredictionInput(BaseModel):
    # Data Dinamis: 24 periode (per 15 hari)
    dynamic_data: List[List[float]] 
    # Data Statis: 6 fitur tanah
    static_data: List[float]

# --- 3. HELPER FUNCTION (Logic Rekomendasi) ---
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

# --- 4. ENDPOINT JSON (ORIGINAL) ---
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

# --- 5. ENDPOINT CSV (BARU - DENGAN AUTO REPAIR) ---
@app.post("/predict-csv")
async def predict_nitrogen_from_csv(
    file: UploadFile = File(...), 
    location_name: str = Form(...)
):
    try:
        # 1. Baca File CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Filter Lokasi
        if 'NAME_3' not in df.columns:
            raise HTTPException(status_code=400, detail="Kolom 'NAME_3' tidak ditemukan di CSV.")
            
        df_loc = df[df['NAME_3'] == location_name].copy()
        
        if df_loc.empty:
            raise HTTPException(status_code=404, detail=f"Lokasi '{location_name}' tidak ditemukan di dataset.")

        # 3. Urutkan & Indexing Waktu
        if 'period_start' not in df_loc.columns:
             raise HTTPException(status_code=400, detail="Kolom 'period_start' wajib ada.")
             
        # Konversi ke datetime
        df_loc['period_start'] = pd.to_datetime(df_loc['period_start'])
        df_loc = df_loc.sort_values('period_start')
        
        # Set tanggal sebagai index agar bisa di-resample
        df_loc.set_index('period_start', inplace=True)

        # --- 4. AUTO-REPAIR: Resampling & Interpolasi ---
        # Memaksa data menjadi frekuensi 15 Harian. 
        # Jika data bolong, baris baru dibuat dengan nilai NaN, lalu diisi interpolasi.
        
        # Ambil hanya kolom numerik untuk diinterpolasi
        numeric_cols = df_loc.select_dtypes(include=[np.number]).columns
        
        # Resample per 15 hari ('15D') -> Interpolasi Linear -> Isi sisa NaN dengan backfill/ffill
        df_resampled = df_loc[numeric_cols].resample('15D').mean().interpolate(method='linear')
        df_resampled = df_resampled.bfill().ffill()

        # Validasi jumlah data setelah perbaikan
        if len(df_resampled) < 24:
             raise HTTPException(status_code=400, detail=f"Data kurang (bahkan setelah interpolasi). Hanya ada {len(df_resampled)} periode 15-harian.")
             
        # Ambil 24 periode terakhir (Window Size ~1 Tahun)
        df_window = df_resampled.tail(24)
        
        # Kembalikan period_start menjadi kolom biasa (untuk info meta)
        df_window = df_window.reset_index()

        # --- 5. Validasi Kolom Fitur ---
        dynamic_cols = ['NDVI', 'EVI', 'OSAVI', 'LST_mean', 'SoilMoisture', 'precip_sum']
        static_cols = ['pH', 'SOC', 'cec', 'clay', 'sand', 'elevation_m']
        
        missing = [c for c in dynamic_cols + static_cols if c not in df_window.columns]
        if missing:
             raise HTTPException(status_code=400, detail=f"Kolom fitur hilang di CSV: {missing}")

        # --- 6. Persiapan Data untuk Model ---
        X_dyn_raw = df_window[dynamic_cols].values 
        X_stat_raw = df_window[static_cols].iloc[-1].values # Ambil statis dari data terakhir

        # Cek NaN terakhir kali (Safety Check)
        if np.isnan(X_dyn_raw).any() or np.isnan(X_stat_raw).any():
             raise HTTPException(status_code=400, detail="Data mengandung nilai NaN yang tidak bisa diperbaiki. Cek kualitas data CSV.")

        # --- 7. Scaling & Prediksi ---
        X_dyn_scaled = sc_dyn.transform(X_dyn_raw)
        X_dyn_final = X_dyn_scaled.reshape(1, 24, 6)

        X_stat_scaled = sc_stat.transform(X_stat_raw.reshape(1, -1))
        X_stat_final = X_stat_scaled

        y_pred_scaled = model.predict([X_dyn_final, X_stat_final])
        y_pred_real = sc_y.inverse_transform(y_pred_scaled).flatten()[0]

        # --- 8. Output ---
        result = generate_recommendation(y_pred_real)
        
        # Info Meta Data
        start_date = df_window['period_start'].iloc[0]
        end_date = df_window['period_start'].iloc[-1]
        days_diff = (end_date - start_date).days

        result['meta'] = {
            "location": location_name,
            "data_period_start": str(start_date.date()),
            "data_period_end": str(end_date.date()),
            "total_days_covered": days_diff,
            "note": "Data telah melalui proses resampling & interpolasi otomatis (15-hari)."
        }
        
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    import uvicorn
    # Jalankan server
    uvicorn.run(app, host="0.0.0.0", port=8000)