"""
shap_service.py
Core logic: load SHAP bundle, compute feature importance dari nilai NPK.
"""
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

from app.config import get_settings
from app.models import FeatureImportance, FertilizerSHAP, SHAPResponse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Bundle loader (singleton via lru_cache)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    """Load fertilizer_shap_bundle.pkl sekali saat startup."""
    settings = get_settings()
    bundle_path = Path(settings.shap_bundle_path)

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"SHAP bundle tidak ditemukan di: {bundle_path}. "
            "Pastikan file fertilizer_shap_bundle.pkl sudah di-mount ke container."
        )

    logger.info(f"Loading SHAP bundle dari {bundle_path} ...")
    try:
        bundle = joblib.load(bundle_path)
    except Exception as joblib_err:
        # Fallback: coba pickle biasa (kadang lebih kompatibel lintas Python versi)
        logger.warning(f"joblib.load gagal ({joblib_err}), coba pickle fallback...")
        import pickle
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
    logger.info("SHAP bundle berhasil di-load.")
    logger.info(f"  cls_target_cols : {bundle['cls_target_cols']}")
    logger.info(f"  cls_input_cols  : {bundle['cls_input_cols']}")
    logger.info(f"  dose_map        : {bundle['dose_map']}")
    return bundle


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def compute_shap(n: float, p: float, k: float) -> SHAPResponse:
    """
    Hitung SHAP feature importance dari nilai N, P, K.

    Parameters
    ----------
    n : float  – Nitrogen (%)
    p : float  – Phosphorus (ppm)
    k : float  – Potassium (ppm)

    Returns
    -------
    SHAPResponse berisi prediksi kelas/dosis & SHAP importance (%)
    """
    bundle = _load_bundle()

    cls_shap_explainers: dict = bundle["cls_shap_explainers"]
    cls_scaler = bundle["cls_scaler_pipeline"]
    cls_input_cols: list[str] = bundle["cls_input_cols"]   # ['N (%)', 'P (ppm)', 'K (ppm)']
    cls_target_cols: list[str] = bundle["cls_target_cols"] # ['UREA', 'SP-36', 'KCL']
    dose_map: dict = bundle["dose_map"]

    # ── Siapkan DataFrame NPK ──────────────────────────────
    npk_raw = {cls_input_cols[0]: n, cls_input_cols[1]: p, cls_input_cols[2]: k}
    npk_df = pd.DataFrame([npk_raw])

    # ── Scale ──────────────────────────────────────────────
    npk_scaled = pd.DataFrame(
        cls_scaler.transform(npk_df),
        columns=cls_input_cols,
    )

    # ── Hitung SHAP per pupuk ──────────────────────────────
    predictions: dict = {}
    shap_results: list[FertilizerSHAP] = []

    for tgt in cls_target_cols:
        if tgt not in cls_shap_explainers:
            logger.warning(f"Explainer untuk {tgt} tidak ada di bundle, skip.")
            continue

        explainer = cls_shap_explainers[tgt]
        sv = explainer.shap_values(npk_scaled)  # list of (1, n_feat) per kelas

        # Normalisasi ke 3D: (n_samples, n_feat, n_cls)
        if isinstance(sv, list):
            sv_3d = np.stack(sv, axis=2)
        else:
            sv_3d = sv  # sudah 3D

        # Expected value → pilih kelas dengan probabilitas tertinggi
        ev = np.array(explainer.expected_value)  # shape (n_cls,)
        pred_cls_idx = int(np.argmax(ev))
        pred_cls = int(explainer.model.n_classes) if hasattr(explainer.model, "n_classes") else len(ev)

        # Cari kelas yang diprediksi dari expected_value (probabilitas tertinggi)
        # TreeExplainer expected_value = base rate tiap kelas
        # Kita ambil kelas dengan total (expected + shap) tertinggi
        shap_sample = sv_3d[0]  # (n_feat, n_cls)
        total_per_cls = ev + shap_sample.sum(axis=0)  # total contribution per kelas
        pred_cls_idx = int(np.argmax(total_per_cls))

        # Mapping idx → label kelas (0,1,2,3,4) dari expected_value length
        n_classes = len(ev)
        # Kelas yang tersedia di model (dari jumlah kelas di expected_value)
        # Kita gunakan dose_map keys yang relevan
        available_classes = sorted(dose_map.keys())[:n_classes]
        pred_class_label = available_classes[pred_cls_idx]
        pred_dose = dose_map.get(int(pred_class_label))

        predictions[tgt] = {
            "class": int(pred_class_label),
            "dose_kg_ha": int(pred_dose) if pred_dose is not None else None,
        }

        # ── Aggregate SHAP importance ──────────────────────
        # mean |SHAP| across semua kelas → aggregate per feature
        mean_agg = np.abs(sv_3d).mean(axis=(0, 2))  # (n_feat,)
        total_abs = float(mean_agg.sum())

        feature_importances: list[FeatureImportance] = []
        for fi, col in enumerate(cls_input_cols):
            shap_val = float(mean_agg[fi])
            pct = (shap_val / total_abs * 100) if total_abs > 0 else 0.0
            feature_importances.append(
                FeatureImportance(
                    feature=col,
                    shap_value=round(shap_val, 6),
                    importance_pct=round(pct, 2),
                )
            )

        # Urutkan descending
        feature_importances.sort(key=lambda x: x.importance_pct, reverse=True)

        shap_results.append(
            FertilizerSHAP(
                fertilizer=tgt,
                predicted_class=int(pred_class_label),
                predicted_dose_kg_ha=int(pred_dose) if pred_dose is not None else None,
                feature_importances=feature_importances,
            )
        )

    return SHAPResponse(
        npk_input=npk_raw,
        predictions=predictions,
        shap=shap_results,
    )


def get_bundle_info() -> dict:
    """Kembalikan metadata bundle (untuk health check)."""
    try:
        bundle = _load_bundle()
        return {
            "status": "loaded",
            "cls_target_cols": bundle["cls_target_cols"],
            "cls_input_cols": bundle["cls_input_cols"],
            "dose_map": bundle["dose_map"],
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
