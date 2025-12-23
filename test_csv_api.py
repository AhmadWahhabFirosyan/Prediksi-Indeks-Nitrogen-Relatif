import requests

# URL endpoint baru
url = "http://localhost:8000/predict-csv"

# Nama file CSV Anda (pastikan file ini ada di folder yang sama)
file_path = "dataset.csv" 

# Lokasi yang ingin dites (harus ada di kolom NAME_3 di CSV)
lokasi = "Arahan"

try:
    # Buka file dalam mode binary ('rb')
    with open(file_path, 'rb') as f:
        # Siapkan payload
        # 'file' adalah nama parameter di FastAPI (file: UploadFile)
        # 'location_name' adalah nama parameter form (location_name: str)
        files = {'file': f}
        data = {'location_name': lokasi}
        
        print(f"Mengirim request untuk lokasi: {lokasi}...")
        
        # Kirim POST request
        response = requests.post(url, files=files, data=data)
        
        # Cek hasil
        if response.status_code == 200:
            print("\n✅ SUKSES!")
            print("Response Server:")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print("\n❌ GAGAL!")
            print(f"Status Code: {response.status_code}")
            print("Pesan Error:", response.text)

except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
except Exception as e:
    print(f"Error: {e}")