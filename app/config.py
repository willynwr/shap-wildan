"""
config.py
Pengaturan aplikasi melalui environment variables.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    port: int = 8000

    # Firebase
    firebase_credentials_path: str = "./firebase-service-account.json"
    firebase_database_url: str = ""
    firebase_database_secret: str = ""  # Legacy DB secret – diutamakan jika diset

    # CORS
    cors_origins: str = "http://localhost:8080,http://localhost:3000"

    # Model
    shap_bundle_path: str = "/app/fertilizer_shap_bundle.pkl"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
