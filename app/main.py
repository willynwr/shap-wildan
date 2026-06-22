"""
main.py
Entry point FastAPI – SHAP Feature Importance Service
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import NPKInput, FirebaseDocInput, SHAPResponse
from app.shap_service import compute_shap, get_bundle_info

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan – pre-load bundle saat startup
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting SHAP API ...")
    info = get_bundle_info()
    if info["status"] == "error":
        logger.error(f"❌ Gagal load SHAP bundle: {info['detail']}")
    else:
        logger.info(f"✅ SHAP bundle loaded: {info}")
    yield
    logger.info("🛑 SHAP API shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="SHAP Feature Importance API",
    description=(
        "FastAPI service untuk menghitung SHAP feature importance "
        "dari nilai NPK hasil prediksi sensor AS7265x. "
        "Digunakan oleh website Laravel untuk menampilkan interpretasi model pupuk."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Kembalikan status service dan info bundle yang ter-load.
    """
    bundle_info = get_bundle_info()
    return {
        "status": "ok" if bundle_info["status"] == "loaded" else "degraded",
        "bundle": bundle_info,
    }


@app.post(
    "/shap/predict",
    response_model=SHAPResponse,
    tags=["SHAP"],
    summary="Hitung SHAP dari nilai NPK",
)
def shap_from_npk(body: NPKInput):
    """
    Hitung SHAP feature importance dari nilai N, P, K.

    **Input** (body JSON):
    ```json
    {
        "N (%)": 2.5,
        "P (ppm)": 18.0,
        "K (ppm)": 120.0
    }
    ```

    Atau menggunakan alias pendek:
    ```json
    {
        "N": 2.5,
        "P": 18.0,
        "K": 120.0
    }
    ```

    **Output**: prediksi kelas pupuk (UREA, SP-36, KCL) beserta dosis dan
    persentase kontribusi tiap fitur (N, P, K) berdasarkan SHAP.
    """
    try:
        result = compute_shap(n=body.N, p=body.P, k=body.K)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Error computing SHAP")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.post(
    "/shap/from-firebase",
    response_model=SHAPResponse,
    tags=["SHAP + Firebase"],
    summary="Hitung SHAP dari data Firebase (by path)",
)
def shap_from_firebase_doc(body: FirebaseDocInput):
    """
    Ambil data NPK dari node Firebase Realtime Database berdasarkan path,
    lalu hitung SHAP feature importance.

    **Input**:
    ```json
    { "doc_path": "predictions/-NxYZ123" }
    ```

    Data di Firebase harus memiliki field: `N`, `P`, `K`
    (atau `N (%)`, `P (ppm)`, `K (ppm)`).
    """
    try:
        from app.firebase_service import get_node
        data = get_node(body.doc_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Firebase credentials error: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Firebase config error: {exc}")
    except Exception as exc:
        logger.exception("Error fetching from Firebase")
        raise HTTPException(status_code=500, detail=f"Firebase error: {exc}")

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Data tidak ditemukan di Firebase path: '{body.doc_path}'",
        )

    # Parse NPK dari data Firebase
    npk = _parse_npk_from_firebase(data, body.doc_path)

    try:
        result = compute_shap(n=npk["N"], p=npk["P"], k=npk["K"])
        # Override predictions dengan dosis aktual dari Firebase (CD/UD/SD)
        firebase_doses = _parse_doses_from_firebase(data)
        result = result.model_copy(update={"predictions": firebase_doses})
        return result
    except Exception as exc:
        logger.exception("Error computing SHAP from Firebase data")
        raise HTTPException(status_code=500, detail=f"SHAP error: {exc}")


@app.get(
    "/shap/firebase-latest",
    response_model=SHAPResponse,
    tags=["SHAP + Firebase"],
    summary="Hitung SHAP dari prediksi terbaru di Firebase",
)
def shap_from_firebase_latest(base_path: str = ""):
    """
    Ambil data terbaru dari node Firebase (berdasarkan push key terbaru),
    lalu hitung SHAP feature importance.

    **Query param**:
    - `base_path` – path koleksi di Firebase (default: `predictions`)

    Contoh: `/shap/firebase-latest?base_path=sensor_results`
    """
    try:
        from app.firebase_service import get_latest_prediction
        snapshot = get_latest_prediction(base_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Firebase credentials error: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Firebase config error: {exc}")
    except Exception as exc:
        logger.exception("Error fetching latest from Firebase")
        raise HTTPException(status_code=500, detail=f"Firebase error: {exc}")

    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada data di Firebase path: '{base_path}'",
        )

    data = snapshot["data"]
    doc_path = f"{base_path}/{snapshot['key']}"
    npk = _parse_npk_from_firebase(data, doc_path)

    try:
        result = compute_shap(n=npk["N"], p=npk["P"], k=npk["K"])
        # Override predictions dengan dosis aktual dari Firebase (CD/UD/SD)
        firebase_doses = _parse_doses_from_firebase(data)
        result = result.model_copy(update={"predictions": firebase_doses})
        return result
    except Exception as exc:
        logger.exception("Error computing SHAP from latest Firebase data")
        raise HTTPException(status_code=500, detail=f"SHAP error: {exc}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse_npk_from_firebase(data: dict, path: str) -> dict:
    """
    Ekstrak nilai N, P, K dari dict Firebase.
    Mendukung key: 'N'/'N (%)', 'P'/'P (ppm)', 'K'/'K (ppm)'.
    """
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=422,
            detail=f"Data di Firebase '{path}' bukan object/dict. Diterima: {type(data)}",
        )

    def _get(keys: list[str], label: str) -> float:
        for k in keys:
            if k in data:
                try:
                    return float(data[k])
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Nilai '{k}' di Firebase tidak bisa dikonversi ke float: {data[k]}",
                    )
        raise HTTPException(
            status_code=422,
            detail=f"Field {label} tidak ditemukan di Firebase '{path}'. "
                   f"Coba salah satu dari: {keys}. Data diterima: {list(data.keys())}",
        )

    return {
        "N": _get(["N", "N (%)", "nitrogen"], "N (Nitrogen)"),
        "P": _get(["P", "P (ppm)", "phosphorus"], "P (Phosphorus)"),
        "K": _get(["K", "K (ppm)", "potassium"], "K (Potassium)"),
    }


def _parse_doses_from_firebase(data: dict) -> dict:
    """
    Ekstrak dosis pupuk aktual dari Firebase:
    - CD (kcl_dose)   → KCL
    - UD (urea_dose)  → UREA
    - SD (sp36_dose)  → SP-36

    Nilai None dikembalikan jika field tidak ada atau tidak bisa dikonversi.
    """
    def safe_int(val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return None

    return {
        "KCL":  {"dose_kg_ha": safe_int(data.get("CD")), "source": "firebase"},
        "UREA": {"dose_kg_ha": safe_int(data.get("UD")), "source": "firebase"},
        "SP-36": {"dose_kg_ha": safe_int(data.get("SD")), "source": "firebase"},
    }
