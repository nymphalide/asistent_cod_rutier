import logging
from typing import List
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase

from app.core.worker_app import task_app # <-- Updated Import
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
from app.schemas.law_unit import LawUnitCreate

logger = logging.getLogger(__name__)

# Singleton Pattern for ML Extractors
GLOBAL_DETERMINISTIC_EXTRACTOR = DeterministicExtractor()
GLOBAL_SEMANTIC_EXTRACTOR = SemanticExtractor()
GLOBAL_CLUSTERING_ENGINE = ClusteringEngine()

@task_app.task(name="ingest_vectors_batch")
async def ingest_vectors_batch_task(unit_ids: List[str]):
    """TASK A: Handles Ollama embeddings and Qdrant (The Heavy Lifter)"""
    logger.info(f"[VECTOR TASK] Woke up! Processing {len(unit_ids)} units.")
    qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    qdrant_repo = QdrantRepository(client=qdrant_client)

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
    finally:
        await qdrant_client.close()


@task_app.task(name="ingest_graph_batch")
async def ingest_graph_batch_task(unit_ids: List[str]):
    """TASK B: Handles GLiNER, HDBSCAN, and Neo4j (The Fast/Volatile Lifter)"""
    logger.info(f"[GRAPH TASK] Woke up! Processing {len(unit_ids)} units.")
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    graph_repo = Neo4jRepository(neo4j_driver)

    try:
        async with AsyncSessionLocal() as pg_session:
            pg_repo = LawUnitRepository(pg_session)
            raw_units = [LawUnitCreate.model_validate(u) for u in [await pg_repo.get(uid) for uid in unit_ids] if u]

            if not raw_units:
                return

            logger.info("[GRAPH TASK] Extracting Knowledge Graph boundaries and semantics...")
            graph_orchestrator = GraphOrchestrator(
                graph_repo=graph_repo,
                deterministic_extractor=GLOBAL_DETERMINISTIC_EXTRACTOR,
                semantic_extractor=GLOBAL_SEMANTIC_EXTRACTOR,
                clustering_engine=GLOBAL_CLUSTERING_ENGINE
            )
            await graph_orchestrator.process_batch(raw_units)
            logger.info(f"[GRAPH TASK] Successfully synced {len(unit_ids)} units to Neo4j.")

    except Exception as e:
        logger.error(f"[GRAPH TASK] Failed: {e}")
        raise
    finally:
        await neo4j_driver.close()