# Use NVIDIA CUDA runtime as base for GPU acceleration
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 1. System dependencies & uv installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install uv

WORKDIR /app

# 2. HEAVY LAYER: Bulletproof PyTorch Install using uv (Core Only)
# --system tells uv to install globally since we are inside a container
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cu121

# 3. LIGHT LAYER: Cached Requirements Install
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# 4. Copy the rest of the application
COPY . .

# 5. Install the developer CLI tool
RUN uv pip install --system -e .

EXPOSE 8000

# Exec ensures Python is PID 1 to receive termination signals
CMD ["exec", "python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]