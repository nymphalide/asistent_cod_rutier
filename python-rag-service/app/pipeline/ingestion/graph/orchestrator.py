import logging
from typing import List

from app.schemas.graph import GraphPayload, LawUnitNode, PartOfEdge
from app.db.graph import Neo4jRepository
from app.pipeline.ingestion.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from app.pipeline.ingestion.graph.clustering import ClusteringEngine
from app.db.models import LawUnit

logger = logging.getLogger(__name__)


class GraphOrchestrator:
    """
    Facade Pattern: Manages the Knowledge Graph ETL workflow.
    Now strictly relies on in-memory domain models, removing circular DB dependencies.
    """

    def __init__(
            self,
            graph_repo: Neo4jRepository,
            deterministic_extractor: DeterministicExtractor,
            semantic_extractor: SemanticExtractor,
            clustering_engine: ClusteringEngine
    ):
        # Strict Dependency Injection Pattern
        self.graph_repo = graph_repo
        self.deterministic_extractor = deterministic_extractor
        self.semantic_extractor = semantic_extractor
        self.clustering_engine = clustering_engine

    async def process_batch(self, units: List[LawUnit]) -> None:
        """
        Executes the full graph formulation pipeline for a batch of pre-loaded LawUnits.
        """
        if not units:
            logger.warning("No valid LawUnits provided to the Orchestrator.")
            return

        logger.info(f"Orchestrating graph extraction for {len(units)} units.")

        # Data Transfer Object Pattern: Master Payload instantiated immediately
        payload = GraphPayload()

        # 1. Process Layer 1: The Skeleton & Deterministic Edges
        for unit in units:
            payload.law_units.append(LawUnitNode(id=unit.id, unit_type=unit.unit_type))

            if unit.parent_id:
                payload.part_of_edges.append(PartOfEdge(child_id=unit.id, parent_id=unit.parent_id))

            # Strategy Pattern execution
            det_result = self.deterministic_extractor.extract_references(unit.id, unit.content)

            payload.reference_edges.extend(det_result.internal_edges)
            payload.external_laws.extend(det_result.external_nodes)
            payload.refers_to_external_edges.extend(det_result.external_edges)

        # 2. Process Layer 2 & 3: Semantic Extraction
        raw_entities = await self.semantic_extractor.extract_batch(units)

        # 3. Entity Resolution & Clustering
        cluster_results = await self.clustering_engine.resolve_entities(raw_entities)

        # 4. Merge ML outputs
        payload.concepts.extend(cluster_results.concepts)
        payload.categories.extend(cluster_results.categories)
        payload.mentions_edges.extend(cluster_results.mentions_edges)
        payload.belongs_to_edges.extend(cluster_results.belongs_to_edges)

        # 5. Load into Knowledge Graph  (Adapter Pattern)
        await self.graph_repo.upsert_payload(payload)
        logger.info(f"Successfully merged graph payload for {len(units)} units into Neo4j.")