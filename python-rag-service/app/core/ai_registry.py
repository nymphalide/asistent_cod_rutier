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

    @staticmethod
    def get_ner_config() -> Dict[str, Any]:
        """Configuration for the Zero-Shot NER extraction."""
        return {
            "model": "urchade/gliner_multi-v2.1",
            "batch_size": 8
        }

    @staticmethod
    def get_clustering_config() -> Dict[str, Any]:
        """Hyperparameters for HDBSCAN and entity resolution."""
        return {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "cosine"
        }