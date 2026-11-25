# NEXORA Production Dockerfile (Single-Server Web Service for Render)
# Runs FastAPI backend serving the compiled React frontend in CPU fallback mode

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CUDA_ENGINE_HOST=0.0.0.0 \
    CUDA_ENGINE_PORT=8000 \
    CUDA_ENGINE_DEVICE=cpu \
    NEXORA_FORCE_FALLBACK=1 \
    CUDA_ENGINE_ENV=production \
    CUDA_ENGINE_DB_PATH=/app/data/training_jobs.db

WORKDIR /app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid downloading multi-gigabyte CUDA binaries on Render
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.4.0,<2.7.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source directories
COPY backend/ /app/backend/
COPY engine/ /app/engine/
COPY custom_kernels/ /app/custom_kernels/
COPY data/ /app/data/
COPY results/ /app/results/
COPY checkpoints/ /app/checkpoints/
COPY frontend/dist/ /app/frontend/dist/
COPY LICENSE NOTICE THIRD_PARTY_LICENSES.md /app/

# Ensure data directory exists with write permissions for SQLite
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Expose default port
EXPOSE 8000

# Start Uvicorn, dynamically binding to Render's PORT environment variable
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
