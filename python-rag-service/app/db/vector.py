import uuid
import logging
from typing import List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.schemas.law_unit import LawUnitEnriched
from app.core.ai_registry import ModelRegistry

logger = logging.getLogger(__name__)

# A static namespace for generating deterministic UUIDs
NAMESPACE_RAG = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


class QdrantRepository:
    """
    Repository layer for interacting with Qdrant.
    Uses Dependency Injection for connection pooling and implements
    strict error handling for distributed transactions.
    """

    def __init__(self, client: AsyncQdrantClient):
        # The client is injected, ensuring we reuse a single connection pool
        self.client = client
        self.collection_name = settings.QDRANT_COLLECTION

    async def initialize_collection(self) -> None:
        try:
            if await self.client.collection_exists(self.collection_name):
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
                return

            dimensions = ModelRegistry.get_embedding_dimensions()

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with {dimensions} dimensions.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise RuntimeError(f"Vector database initialization failed: {e}") from e

    def _map_to_points(self, unit: LawUnitEnriched) -> List[models.PointStruct]:
        points = []

        if unit.content_vector:
            content_id = str(uuid.uuid5(NAMESPACE_RAG, f"{unit.id}_content"))
            points.append(
                models.PointStruct(
                    id=content_id,
                    vector=unit.content_vector,
                    payload={
                        "unit_id": unit.id,
                        "unit_type": unit.unit_type.value,
                        "vector_type": "content",
                        "content": unit.content,
                        "parent_id": unit.parent_id
                    }
                )
            )

        if unit.question_vectors and unit.hypothetical_questions:
            for idx, (q_vector, q_text) in enumerate(zip(unit.question_vectors, unit.hypothetical_questions)):
                question_id = str(uuid.uuid5(NAMESPACE_RAG, f"{unit.id}_question_{idx}"))
                points.append(
                    models.PointStruct(
                        id=question_id,
                        vector=q_vector,
                        payload={
                            "unit_id": unit.id,
                            "unit_type": unit.unit_type.value,
                            "vector_type": "question",
                            "question_text": q_text,
                            "parent_id": unit.parent_id
                        }
                    )
                )

        return points

    async def bulk_upsert(self, units: List[LawUnitEnriched]) -> None:
        if not units:
            return

        all_points = []
        for unit in units:
            all_points.extend(self._map_to_points(unit))

        if not all_points:
            return

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=all_points
            )
            logger.info(f"Successfully upserted {len(all_points)} points to Qdrant.")
        except Exception as e:
            logger.error(f"Qdrant upsert failed. Rolling back transaction context. Error: {e}")
            # Raising the error is critical so the Orchestrator knows to abort the Postgres save
            raise