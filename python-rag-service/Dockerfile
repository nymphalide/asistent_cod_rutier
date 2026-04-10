# Use NVIDIA CUDA runtime as base for GPU acceleration
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. HEAVY LAYER: Install PyTorch FIRST and cache this layer
# We remove --no-cache-dir temporarily so if it fails, pip can resume the download
# increased default timeout prevents pip from dropping large file downloads
RUN pip3 install --default-timeout=1000 torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. LIGHT LAYER: Copy requirements and install the rest
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application
COPY . .

# Port for FastAPI
EXPOSE 8000

# Default command (will be overridden by Docker Compose for the worker)
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]