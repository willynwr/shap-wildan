# ─────────────────────────────────────────────────────────────
# Stage 1: Builder – install dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /build

# Install build tools yang mungkin dibutuhkan oleh beberapa package (shap, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────
FROM python:3.9-slim AS runtime

LABEL maintainer="SHAP API Service"
LABEL description="FastAPI service untuk SHAP feature importance pupuk"

WORKDIR /app

# Copy installed packages dari builder
COPY --from=builder /install /usr/local

# Copy source code
COPY app/ ./app/

# Copy model bundle (akan di-override oleh volume mount jika ada)
COPY fertilizer_shap_bundle.pkl ./fertilizer_shap_bundle.pkl

# Environment defaults (dapat di-override via .env atau docker-compose)
ENV APP_ENV=production \
    PORT=8000 \
    SHAP_BUNDLE_PATH=/app/fertilizer_shap_bundle.pkl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Non-root user untuk keamanan
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
