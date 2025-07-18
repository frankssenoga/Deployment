# ───────────────────────────────────────────────
#  Dockerfile  for “Virtual Machine Attack Predictor”
# ───────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set work directory inside the container
WORKDIR /app

# Optional: install system packages required by SHAP or matplotlib (for SHAP plots)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files (app.py, templates/, static/, models, etc.)
COPY . .

# Expose port (useful for local Docker runs, Render uses ENV PORT)
EXPOSE 5000

# Launch the app using Gunicorn
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000}"]
