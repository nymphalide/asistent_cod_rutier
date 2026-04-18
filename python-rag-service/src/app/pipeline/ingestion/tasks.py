import logging
import signal
import asyncio
import sys
from typing import List
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase # type: ignore

from src.app.core.worker_app import task_app
from src.app.core.config import settings
from src.app.db.session import AsyncSessionLocal
from src.app.db.repository import LawUnitRepository
from src.app.db.vector import QdrantRepository
from src.app.db.graph import Neo4jRepository
from src.app.pipeline.ingestion.enricher import EnricherService
from src.app.pipeline.ingestion.graph.orchestrator import GraphOrchestrator
from src.app.pipeline.ingestion.graph.extractors.deterministic import DeterministicExtractor
from src.app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from src.app.pipeline.ingestion.graph.clustering import ClusteringEngine
from src.app.schemas.law_unit import LawUnitCreate

logger = logging.getLogger(__name__)

# Singleton Pattern for Database Clients (Connection Pooling)
GLOBAL_QDRANT_CLIENT = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
GLOBAL_NEO4J_DRIVER = AsyncGraphDatabase.driver(
    settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)


@task_app.task(name="ingest_vectors_batch")
async def ingest_vectors_batch_task(unit_ids: List[str]):
    """TASK A: Handles Ollama embeddings and Qdrant (The Heavy Lifter)"""
    logger.info(f"[VECTOR TASK] Woke up! Processing {len(unit_ids)} units.")
    qdrant_repo = QdrantRepository(client=GLOBAL_QDRANT_CLIENT)

    try:
        async with AsyncSessionLocal() as pg_session:
            pg_repo = LawUnitRepository(pg_session)
            raw_units = [LawUnitCreate.model_validate(u) for u in [await pg_repo.get(uid) for uid in unit_ids] if u]

            if not raw_units:
                return

            enricher = EnricherService()
            logger.info("[VECTOR TASK] Enriching data with Embeddings and Questions...")
            enriched_units = await enricher.enrich_batch(raw_units)

            logger.info("[VECTOR TASK] Upserting vectors to Qdrant...")
            await qdrant_repo.initialize_collection()
            await qdrant_repo.bulk_upsert(enriched_units)
            logger.info(f"[VECTOR TASK] Successfully synced {len(unit_ids)} units to Qdrant.")

    except Exception as e:
        logger.error(f"[VECTOR TASK] Failed: {e}")
        raise


@task_app.task(name="ingest_graph_batch")
async def ingest_graph_batch_task(unit_ids: List[str]):
    """TASK B: Handles GLiNER, HDBSCAN, and Neo4j (The Fast/Volatile Lifter)"""
    logger.info(f"[GRAPH TASK] Woke up! Processing {len(unit_ids)} units.")
    graph_repo = Neo4jRepository(GLOBAL_NEO4J_DRIVER)

    try:
        async with AsyncSessionLocal() as pg_session:
            pg_repo = LawUnitRepository(pg_session)
            raw_units = [LawUnitCreate.model_validate(u) for u in [await pg_repo.get(uid) for uid in unit_ids] if u]

            if not raw_units:
                return

            logger.info("[GRAPH TASK] Extracting Knowledge Graph boundaries and semantics...")

            # Instantiating locally. SingletonMeta ensures memory is shared efficiently.
            graph_orchestrator = GraphOrchestrator(
                graph_repo=graph_repo,
                deterministic_extractor=DeterministicExtractor(),
                semantic_extractor=SemanticExtractor(),
                clustering_engine=ClusteringEngine()
            )
            await graph_orchestrator.process_batch(raw_units)
            logger.info(f"[GRAPH TASK] Successfully synced {len(unit_ids)} units to Neo4j.")

    except Exception as e:
        logger.error(f"[GRAPH TASK] Failed: {e}")
        raise


async def shutdown_connections():
    """Gracefully closes global connection pools."""
    logger.info("Worker shutting down. Closing database connection pools...")

    try:
        if GLOBAL_QDRANT_CLIENT:
            await GLOBAL_QDRANT_CLIENT.close()
            logger.info("Qdrant pool closed.")

        if GLOBAL_NEO4J_DRIVER:
            await GLOBAL_NEO4J_DRIVER.close()
            logger.info("Neo4j pool closed.")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def handle_sigterm(signum, frame):
    """Catches termination signals cross-platform."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(shutdown_connections())
        loop.call_later(15.0, sys.exit, 0)
    except RuntimeError:
        sys.exit(0)


signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)