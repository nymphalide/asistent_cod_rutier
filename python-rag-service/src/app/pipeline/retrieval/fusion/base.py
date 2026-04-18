from abc import ABC, abstractmethod
from typing import List
from src.app.schemas.retrieval import RetrievalRequest, RetrievedChunk

class BaseFusionEngine(ABC):
    """
    Strategy Pattern for the Fusion layer.
    """
    @abstractmethod
    async def fuse_and_rank(self, request: RetrievalRequest, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Applies RRF, runs the Cross-Encoder, and prunes low-scoring chunks.
        """
        pass