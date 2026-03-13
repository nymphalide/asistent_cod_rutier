import logging
from typing import List

from app.schemas.graph import (
    GraphPayload,
    LawUnitNode,
    PartOfEdge
)
from app.db.repository import LawUnitRepository
from app.db.graph import Neo4jRepository
from app.pipeline.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.graph.extractors.semantic import SemanticExtractor
from app.pipeline.graph.resolution.clustering import ClusteringEngine

logger = logging.getLogger(__name__)


class GraphOrchestrator:
    """
    Facade Pattern: Manages the Knowledge Graph ETL workflow.
    Orchestrates the handoff between PostgreSQL, ML Extractors, and Neo4j.
    """

    def __init__(
            self,
            pg_repo: LawUnitRepository,
            graph_repo: Neo4jRepository,
            deterministic_extractor: DeterministicExtractor,
            semantic_extractor: SemanticExtractor,
            clustering_engine: ClusteringEngine
    ):
        # Dependency Injection Pattern ensures loose coupling and testability
        self.pg_repo = pg_repo
        self.graph_repo = graph_repo
        self.deterministic_extractor = deterministic_extractor
        self.semantic_extractor = semantic_extractor
        self.clustering_engine = clustering_engine

    async def process_batch(self, unit_ids: List[str]) -> None:
        """
        Executes the full graph formulation pipeline for a batch of LawUnits.
        """
        logger.info(f"Orchestrating graph extraction for {len(unit_ids)} units.")

        # 1. Fetch Source Data (Repository Pattern)
        units = []
        for uid in unit_ids:
            unit = await self.pg_repo.get(uid)
            if unit:
                units.append(unit)

        if not units:
            logger.warning("No valid LawUnits found for provided IDs.")
            return

        # Data Transfer Object Pattern: Master Payload instantiated immediately
        payload = GraphPayload()

        # 2. Process Layer 1: The Skeleton & Deterministic Edges
        for unit in units:
            # Map Postgres entity to lightweight Graph DTO
            payload.law_units.append(LawUnitNode(id=unit.id, unit_type=unit.unit_type))

            # Reconstruct the structural hierarchy
            if unit.parent_id:
                payload.part_of_edges.append(PartOfEdge(child_id=unit.id, parent_id=unit.parent_id))

            # Execute the Strategy Pattern for regex parsing
            det_result = self.deterministic_extractor.extract_references(unit.id, unit.content)

            # Unpack the deterministic DTO into the master payload
            payload.reference_edges.extend(det_result.internal_edges)
            payload.external_laws.extend(det_result.external_nodes)
            payload.refers_to_external_edges.extend(det_result.external_edges)

        # 3. Process Layer 2 & 3: Semantic Extraction
        raw_entities = await self.semantic_extractor.extract_batch(units)

        # 4. Entity Resolution & Clustering
        cluster_results = await self.clustering_engine.resolve_entities(raw_entities)

        # 5. Merge ML outputs into the master payload
        payload.concepts.extend(cluster_results.concepts)
        payload.categories.extend(cluster_results.categories)
        payload.mentions_edges.extend(cluster_results.mentions_edges)
        payload.belongs_to_edges.extend(cluster_results.belongs_to_edges)

        # 6. Load into Graph Database (Adapter Pattern execution)
        await self.graph_repo.upsert_payload(payload)
        logger.info(f"Successfully merged graph payload for {len(units)} units into Neo4j.")