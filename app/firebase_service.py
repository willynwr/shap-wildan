"""
firebase_service.py
Koneksi ke Firebase Realtime Database.

Strategi auth (urutan prioritas):
  1. FIREBASE_DATABASE_SECRET  → REST API langsung (tidak perlu file JSON)
  2. FIREBASE_CREDENTIALS_PATH → firebase-admin SDK dengan service account JSON
"""
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_firebase_initialized = False  # hanya digunakan untuk mode service-account


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _get_settings():
    from app.config import get_settings
    return get_settings()


def _use_secret_mode() -> bool:
    """Return True jika FIREBASE_DATABASE_SECRET tersedia."""
    return bool(_get_settings().firebase_database_secret)


def _rest_url(path: str) -> str:
    """Bangun URL REST Firebase untuk path tertentu."""
    settings = _get_settings()
    base = settings.firebase_database_url.rstrip("/")
    clean_path = path.strip("/")
    return f"{base}/{clean_path}.json"


def _rest_get(path: str, params: Optional[dict] = None) -> requests.Response:
    """GET ke Firebase REST API menggunakan database secret sebagai auth."""
    settings = _get_settings()
    if not settings.firebase_database_url:
        raise ValueError(
            "FIREBASE_DATABASE_URL belum diset di .env. "
            "Contoh: https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com"
        )
    qparams = {"auth": settings.firebase_database_secret}
    if params:
        qparams.update(params)
    url = _rest_url(path)
    resp = requests.get(url, params=qparams, timeout=10)
    resp.raise_for_status()
    return resp


# ─────────────────────────────────────────────────────────────
# Mode service-account (fallback)
# ─────────────────────────────────────────────────────────────

def _init_firebase():
    """Inisialisasi Firebase Admin SDK (sekali saja) – hanya dipakai jika tidak ada secret."""
    global _firebase_initialized
    if _firebase_initialized:
        return

    import firebase_admin
    from firebase_admin import credentials
    from pathlib import Path

    settings = _get_settings()
    cred_path = Path(settings.firebase_credentials_path)
    db_url = settings.firebase_database_url

    if not cred_path.exists():
        raise FileNotFoundError(
            f"Firebase service account tidak ditemukan: {cred_path}. "
            "Letakkan file JSON credentials atau set FIREBASE_DATABASE_SECRET di .env"
        )

    if not db_url:
        raise ValueError(
            "FIREBASE_DATABASE_URL belum diset di .env. "
            "Contoh: https://YOUR_PROJECT_ID-default-rtdb.firebaseio.com"
        )

    cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred, {"databaseURL": db_url})
    _firebase_initialized = True
    logger.info(f"Firebase initialized (service account) → {db_url}")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def get_node(path: str) -> Optional[dict]:
    """
    Ambil data dari Firebase Realtime Database berdasarkan path node.

    Parameters
    ----------
    path : str  Contoh: "predictions/latest" atau "predictions/-NxYZ123"

    Returns
    -------
    dict | None
    """
    if _use_secret_mode():
        logger.debug(f"[REST] GET {path}")
        resp = _rest_get(path)
        data = resp.json()
        if data is None:
            logger.warning(f"Node '{path}' tidak ditemukan atau kosong di Firebase.")
        return data
    else:
        _init_firebase()
        from firebase_admin import db
        ref = db.reference(path)
        data = ref.get()
        if data is None:
            logger.warning(f"Node '{path}' tidak ditemukan atau kosong di Firebase.")
        return data


def get_latest_prediction(base_path: str = "") -> Optional[dict]:
    """
    Ambil data terbaru dari node Firebase.

    Mendukung dua mode otomatis:
    - **Flat realtime node**: data langsung di node (overwrite tiap kiriman sensor).
      Contoh: willy/ → {N: 1.2, P: 3.4, K: 5.6, ...}
    - **Koleksi push-key**: tiap kiriman disimpan sebagai child baru.
      Contoh: predictions/ → {"-NxYZ": {N:..}, "-NxAB": {N:..}}

    Parameters
    ----------
    base_path : str  Path ke node Firebase, default "" (root URL)

    Returns
    -------
    dict | None  Berisi 'key' dan 'data'
    """
    if _use_secret_mode():
        logger.debug(f"[REST] GET from '{base_path}'")
        resp = _rest_get(base_path)
        snapshot = resp.json()

        if not snapshot or not isinstance(snapshot, dict):
            logger.warning(f"Tidak ada data di node '{base_path}'.")
            return None

        # Deteksi flat node vs koleksi push-key
        has_nested = any(isinstance(v, dict) for v in snapshot.values())
        if not has_nested:
            # Flat realtime node – data langsung tersedia
            logger.info(f"Flat realtime node terdeteksi di '{base_path}'")
            return {"key": "realtime", "data": snapshot}

        # Koleksi push-key – ambil entry terbaru
        resp2 = _rest_get(
            base_path,
            params={"orderBy": '"$key"', "limitToLast": 1}
        )
        snapshot2 = resp2.json()
        if not snapshot2:
            logger.warning(f"Tidak ada data di koleksi '{base_path}'.")
            return None
        key, data = next(iter(snapshot2.items()))
        logger.info(f"Latest prediction key: {key}")
        return {"key": key, "data": data}
    else:
        _init_firebase()
        from firebase_admin import db
        ref = db.reference(base_path) if base_path else db.reference("/")
        snapshot = ref.get()

        if not snapshot or not isinstance(snapshot, dict):
            logger.warning(f"Tidak ada data di node '{base_path}'.")
            return None

        has_nested = any(isinstance(v, dict) for v in snapshot.values())
        if not has_nested:
            logger.info(f"Flat realtime node terdeteksi di '{base_path}'")
            return {"key": "realtime", "data": snapshot}

        snapshot2 = ref.order_by_key().limit_to_last(1).get()
        if not snapshot2:
            logger.warning(f"Tidak ada data di koleksi '{base_path}'.")
            return None
        key, data = next(iter(snapshot2.items()))
        logger.info(f"Latest prediction key: {key}")
        return {"key": key, "data": data}
