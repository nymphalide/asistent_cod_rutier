import torch
from fastapi import FastAPI
from app.core.config import settings
from app.db import engine, Base
from app.db import models
app = FastAPI(title=settings.PROJECT_NAME)

@app.get("/health")
async def health_check():
    """
    Checks service status and hardware acceleration.
    """
    return {
        "status": "online",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }


@app.on_event("startup")
async def startup_event():
    print(f"🚀 {settings.PROJECT_NAME} starting on {settings.DEVICE.upper()}")

    # --- THIS IS THE MISSING PIECE ---
    # It creates the tables defined in models.py in the Postgres DB
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database schema initialized (Tables created).")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

    print(f"📍 Database: {settings.DATABASE_URL.split('@')[-1]}")
    print(f"📍 Vector Store: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")