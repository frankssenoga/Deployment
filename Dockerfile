# ──────────────────────────────────────────────
# 1. Base image
# ──────────────────────────────────────────────
# Use the latest slim Python image (tiny footprint, Debian-based)
FROM python:3.11-slim

# ──────────────────────────────────────────────
# 2. Environment settings
# ──────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ──────────────────────────────────────────────
# 3. OS-level deps (minimal build tooling)
# ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────
# 4. Working directory
# ──────────────────────────────────────────────
WORKDIR /app

# ──────────────────────────────────────────────
# 5. Python deps
# ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ──────────────────────────────────────────────
# 6. App source + model/scaler
# ──────────────────────────────────────────────
COPY . .

# ──────────────────────────────────────────────
# 7. Expose (Render sets $PORT at runtime, but this aids local tests)
# ──────────────────────────────────────────────
EXPOSE 5000

# ──────────────────────────────────────────────
# 8. Start the service
# ──────────────────────────────────────────────
# Render injects $PORT; locally defaults to 5000
CMD ["bash", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000}"]
