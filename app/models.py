"""
models.py
Pydantic schemas untuk request & response FastAPI.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class NPKInput(BaseModel):
    """
    Input nilai N, P, K yang sudah dihasilkan oleh model regresi di Raspberry Pi.
    Nilai ini yang dikirim ke Firebase dan diterima oleh FastAPI untuk dihitung SHAP-nya.
    """
    N: float = Field(..., alias="N (%)", description="Nitrogen (%)")
    P: float = Field(..., alias="P (ppm)", description="Phosphorus (ppm)")
    K: float = Field(..., alias="K (ppm)", description="Potassium (ppm)")

    model_config = {"populate_by_name": True}


class FirebaseDocInput(BaseModel):
    """Input document key/path di Firebase Realtime Database."""
    doc_path: str = Field(
        ...,
        description="Path ke node Firebase, contoh: 'predictions/latest' atau 'predictions/-NxYZ123'",
    )


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    """Persentase feature importance SHAP untuk satu fitur."""
    feature: str
    shap_value: float = Field(..., description="Raw SHAP value (aggregated)")
    importance_pct: float = Field(..., description="Persentase kontribusi (0-100)")


class FertilizerSHAP(BaseModel):
    """Hasil SHAP analysis untuk satu jenis pupuk (hanya feature importance)."""
    fertilizer: str
    feature_importances: list[FeatureImportance]


class SHAPResponse(BaseModel):
    """Response lengkap dari endpoint SHAP."""
    # Input yang digunakan
    npk_input: dict = Field(..., description="Nilai N, P, K yang menjadi input")

    # Hasil prediksi kelas & dosis per pupuk
    predictions: dict = Field(
        ...,
        description="Prediksi kelas & dosis: {'UREA': {'class': 2, 'dose_kg_ha': 75}, ...}",
    )

    # SHAP feature importance per pupuk
    shap: list[FertilizerSHAP] = Field(
        ...,
        description="SHAP feature importance per fertilizer",
    )

    # Metadata
    model_version: str = "fertilizer_shap_bundle_v1"
