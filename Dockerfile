# ───────────────────────────────────────────────
#  Dockerfile  for “Virtual Machine Attack Predictor”
# ───────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Workdir inside the container
WORKDIR /app

# Install system packages (optional: remove if slim image works without)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && \
#     rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (app.py, templates/, static/, *.pkl, etc.)
COPY . .

# Expose port for local Docker runs (Render injects $PORT automatically)
EXPOSE 5000

# Start the service
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
