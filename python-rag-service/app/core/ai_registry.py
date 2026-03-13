from typing import Dict, Any


class ModelRegistry:
    """
    Centralized registry for AI model hyperparameters.
    Separates behavioral AI logic from infrastructure environment variables.
    """

    @staticmethod
    def get_enricher_chat_config() -> Dict[str, Any]:
        """Configuration for the QuOTE strategy question generation."""
        return {
            "model": "llama3.1:8b",
            "format": "json",
            "options": {
                "temperature": 0.3,  # Slight variance to get distinct questions
                "top_p": 0.9,
            }
        }

    @staticmethod
    def get_embedding_model() -> str:
        """The designated embedding model for dense vectors."""
        return "nomic-embed-text"

    @staticmethod
    def get_embedding_dimensions() -> int:
        """Required by Qdrant to initialize the vector index."""
        return 768  # nomic-embed-text outputs 768 dimensions