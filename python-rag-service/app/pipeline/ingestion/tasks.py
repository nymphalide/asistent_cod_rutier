import logging
from typing import List
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase

from app.db.queue import task_app
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.repository import LawUnitRepository
from app.db.vector import QdrantRepository
from app.db.graph import Neo4jRepository
from app.pipeline.ingestion.enricher import EnricherService
from app.pipeline.ingestion.graph.orchestrator import GraphOrchestrator
from app.pipeline.ingestion.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from app.pipeline.ingestion.graph.clustering import ClusteringEngine

logger = logging.getLogger(__name__)


@task_app.task(name="process_knowledge_batch")
async def process_knowledge_batch_task(unit_ids: List[str]):
    """
    Worker Pattern: Handles all ML and network I/O operations asynchronously.
    """
    logger.info(f"Worker woke up! Processing batch of {len(unit_ids)} units.")

    # Connection Pool Pattern: Initialize isolated external connections
    qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    qdrant_repo = QdrantRepository(client=qdrant_client)

    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    graph_repo = Neo4jRepository(neo4j_driver)

    try:
        # 1. Fetch data from the relational anchor exactly once
        async with AsyncSessionLocal() as pg_session:
            pg_repo = LawUnitRepository(pg_session)

            raw_units = []
            for uid in unit_ids:
                unit = await pg_repo.get(uid)
                if unit:
                    raw_units.append(unit)

            if not raw_units:
                logger.warning("No units found in Postgres. Aborting task.")
                return

            # 2. Enrich for Qdrant (QuOTE vectors)
            enricher = EnricherService()
            logger.info("Enriching data with Embeddings and Questions...")
            enriched_units = await enricher.enrich_batch(raw_units)

            logger.info("Upserting vectors to Qdrant...")
            await qdrant_repo.initialize_collection()
            await qdrant_repo.bulk_upsert(enriched_units)

            # 3. Extract for Neo4j (Passing the in-memory raw_units directly)
            logger.info("Extracting Knowledge Graph boundaries and semantics...")
            graph_orchestrator = GraphOrchestrator(
                graph_repo=graph_repo,
                deterministic_extractor=DeterministicExtractor(),
                semantic_extractor=SemanticExtractor(),
                clustering_engine=ClusteringEngine()
            )
            await graph_orchestrator.process_batch(raw_units)

        logger.info(f"Batch of {len(unit_ids)} units successfully processed and synced.")

    except Exception as e:
        logger.error(f"Worker task failed: {e}")
        raise  # Triggers the Retry Pattern in Procrastinate

    finally:
        await qdrant_client.close()
        await neo4j_driver.close()