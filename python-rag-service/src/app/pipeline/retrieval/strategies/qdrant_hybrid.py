import logging
from typing import List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.app.core.config import settings
from src.app.core.ai_registry import ModelRegistry
from src.app.clients.llm_gateway import LLMGateway
from src.app.pipeline.retrieval.strategies.base import BaseRetrievalStrategy
from src.app.schemas.retrieval import RetrievalRequest, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantHybridStrategy(BaseRetrievalStrategy):
    """
    Executes a native Hybrid Search (Dense + Sparse) in Qdrant.
    Implements the Strategy Pattern contract.
    """

    def __init__(self, client: AsyncQdrantClient, llm_gateway: LLMGateway):
        # We inject the singleton clients so we don't recreate pools
        self.client = client
        self.gateway = llm_gateway
        self.collection_name = settings.QDRANT_COLLECTION
        self.dense_model = ModelRegistry.get_embedding_model()

        # We need the sparse model locally just for querying, exactly like in Ingestion
        from fastembed import SparseTextEmbedding
        self.sparse_model = SparseTextEmbedding(model_name=ModelRegistry.get_sparse_embedding_model())

    @property
    def strategy_name(self) -> str:
        return "qdrant_hybrid"

    async def retrieve(self, request: RetrievalRequest) -> List[RetrievedChunk]:
        if not request.query_text:
            logger.warning("Qdrant strategy called without query_text. Returning empty.")
            return []

        try:
            # 1. Generate Query Vectors (Dense via Ollama, Sparse via FastEmbed CPU)
            dense_vector = await self.gateway.get_embedding(request.query_text, self.dense_model)

            # fastembed is synchronous, so we run it quickly on the main thread
            # (or use asyncio.to_thread if you notice loop blocking)
            sparse_result = list(self.sparse_model.query_embed(request.query_text))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist()
            )

            # 2. Build Pre-filters (Optional, if DSPy decided to filter by category)
            filter_conditions = []
            if request.metadata_filters:
                for key, value in request.metadata_filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )

            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

            # 3. Execute Native Hybrid Search
            # We use 'prefetch' to let Qdrant do the RRF fusion internally
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_filter=query_filter,
                limit=request.top_k,
                with_payload=True,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="",  # Default dense vector
                        limit=request.top_k * 2,  # Fetch more candidates before fusing
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="text-sparse",
                        limit=request.top_k * 2,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF)
            )

            # 4. Map to Standardized DTO Contract
            retrieved_chunks = []
            for hit in search_results:
                payload = hit.payload or {}
                retrieved_chunks.append(
                    RetrievedChunk(
                        unit_id=payload.get("unit_id", "unknown"),
                        content=payload.get("content") or payload.get("question_text", ""),
                        score=hit.score,
                        source_strategy=self.strategy_name,
                        parent_id=payload.get("parent_id")
                    )
                )

            logger.info(f"Qdrant Hybrid returned {len(retrieved_chunks)} chunks.")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Qdrant retrieval failed: {e}")
            # Catch gracefully to prevent crashing the orchestrator
            return []