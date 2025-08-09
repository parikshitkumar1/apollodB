# Use slim Python image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    UVICORN_WORKERS=1 \
    MPLBACKEND=Agg

# System dependencies for audio and plotting backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libgomp1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first for better build caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

# Copy source (backend, web, models, assets)
COPY . .

# Cloud Run provides $PORT; default to 8080 for local
ENV PORT=8080

# Expose for local usage (Cloud Run ignores EXPOSE)
EXPOSE 8080

# Start uvicorn
CMD ["bash", "-lc", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]
