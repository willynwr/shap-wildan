# SHAP Feature Importance API

FastAPI service untuk menghitung SHAP feature importance dari hasil prediksi NPK sensor AS7265x. Digunakan oleh website Laravel untuk menampilkan interpretasi model pupuk.

---

## Arsitektur

```
[Raspberry Pi] ─→ Firebase Realtime DB
                         ↓
[Laravel Website] ──── HTTP ──→ [FastAPI SHAP API :8000]
                                       ↓
                            Load fertilizer_shap_bundle.pkl
                            Hitung SHAP per pupuk (UREA/SP-36/KCL)
                            Return JSON: % feature importance
                                       ↓
                 ←─────── JSON Response ──────────────────
```

---

## Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET`  | `/health` | Health check & status bundle |
| `POST` | `/shap/predict` | Input NPK langsung → SHAP importance |
| `POST` | `/shap/from-firebase` | Input Firebase path → ambil NPK → SHAP |
| `GET`  | `/shap/firebase-latest` | Ambil entry terbaru Firebase → SHAP |

---

## Quick Start (Docker)

### 1. Buat file `.env`

```bash
cp .env.example .env
```

Edit `.env`:
```env
FIREBASE_CREDENTIALS_PATH=./firebase-service-account.json
FIREBASE_DATABASE_URL=https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com
CORS_ORIGINS=http://localhost:8080
SHAP_BUNDLE_PATH=/app/fertilizer_shap_bundle.pkl
```

### 2. Letakkan Firebase credentials

```bash
cp /path/to/your-firebase-service-account.json ./firebase-service-account.json
```

### 3. Build & Run

```bash
docker compose up --build -d
```

### 4. Cek status

```bash
docker compose logs -f
curl http://localhost:8000/health
```

### 5. Swagger UI

Buka browser: http://localhost:8000/docs

---

## Contoh Request dari Laravel

### POST /shap/predict

```php
// Laravel HTTP Client
$response = Http::post('http://localhost:8000/shap/predict', [
    'N (%)' => 2.5,
    'P (ppm)' => 18.0,
    'K (ppm)' => 120.0,
]);

$data = $response->json();
```

### Contoh Response

```json
{
  "npk_input": {
    "N (%)": 2.5,
    "P (ppm)": 18.0,
    "K (ppm)": 120.0
  },
  "predictions": {
    "UREA":  { "class": 2, "dose_kg_ha": 75 },
    "SP-36": { "class": 1, "dose_kg_ha": 50 },
    "KCL":   { "class": 3, "dose_kg_ha": 100 }
  },
  "shap": [
    {
      "fertilizer": "UREA",
      "predicted_class": 2,
      "predicted_dose_kg_ha": 75,
      "feature_importances": [
        { "feature": "N (%)",   "shap_value": 0.361672, "importance_pct": 88.24 },
        { "feature": "K (ppm)", "shap_value": 0.038258, "importance_pct": 9.33  },
        { "feature": "P (ppm)", "shap_value": 0.009941, "importance_pct": 2.43  }
      ]
    },
    ...
  ],
  "model_version": "fertilizer_shap_bundle_v1"
}
```

### GET /shap/firebase-latest

```php
$response = Http::get('http://localhost:8000/shap/firebase-latest', [
    'base_path' => 'predictions'
]);
```

---

## Development (tanpa Docker)

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

---

## Struktur Data Firebase

Node Firebase harus mengandung field:

```json
{
  "N": 2.5,
  "P": 18.0,
  "K": 120.0
}
```

Juga mendukung nama panjang: `N (%)`, `P (ppm)`, `K (ppm)`.

---

## Docker Commands

```bash
# Build ulang setelah perubahan kode
docker compose up --build -d

# Lihat logs
docker compose logs -f shap-api

# Stop
docker compose down

# Restart
docker compose restart shap-api
```
