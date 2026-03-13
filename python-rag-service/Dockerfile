# Use NVIDIA CUDA runtime as base for GPU acceleration
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts and ensure Python output is sent to logs immediately
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Merged and sorted system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Merged Python package installation:
# 1. Install PyTorch with specific CUDA 12.1 support
# 2. Install remaining RAG dependencies
RUN pip3 install --no-cache-dir \
    torch \
    torchaudio \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the entire Python source
# Ensure main.py exists in this directory before building
COPY . .

# Port for FastAPI
EXPOSE 8000

# Start the application using Uvicorn 
# We use the module flag to ensure main.py is found within /app
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]