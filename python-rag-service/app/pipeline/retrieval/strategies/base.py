from abc import ABC, abstractmethod
from typing import List

from app.schemas.retrieval import RetrievalRequest, RetrievedChunk


class BaseRetrievalStrategy(ABC):
    """
    The Strategy Pattern interface.
    Every specific database adapter MUST implement this contract.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Returns the identifier for this strategy (e.g., 'qdrant_dense')."""
        pass

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> List[RetrievedChunk]:
        """
        Executes the search logic specific to the underlying database.
        Must catch its own DB-specific exceptions and return an empty list on failure.
        """
        pass