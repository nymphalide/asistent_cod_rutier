import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, validator
from typing import Optional


class Settings(BaseSettings):
    """
    Configuration for the Python RAG microservice. [cite: 112]
    This class VALIDATES the environment; it does not set the truth.
    """
    PROJECT_NAME: str = "Optimised Romanian Traffic Code RAG"
    # --- File Paths ---
    RAW_DATA_DIR: str = "data/raw_text"

    # --- Databases (No defaults here forces them to be in .env) ---
    DATABASE_URL: str = Field(validation_alias="PYTHON_DATABASE_URL")
    QDRANT_HOST: str  # Required
    QDRANT_PORT: int  # Required
    QDRANT_COLLECTION: str = "traffic_code_vectors"

    # Graph Store is optional in the plan [cite: 15]
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None


    # --- Local AI Settings (Ollama) ---
    # WSL2 usually resolves localhost fine, but if Docker bridge fails later, use host.docker.internal
    OLLAMA_HOST: str = "http://localhost:11434"
    MAX_CONCURRENT_ENRICHMENT_TASKS: int = 5

    # --- Hardware ---
    USE_CUDA: bool = True

    @property
    def DEVICE(self) -> str:
        """Dynamically selects the GPU (RTX 4060) if available."""
        if self.USE_CUDA and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # Crucial: This tells Pydantic WHERE the truth is.
    model_config = SettingsConfigDict(
        # Pydantic reads left-to-right.
        # It loads .env first, then overwrites anything it finds in .env.local
        env_file=("../.env", "../.env.local"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()