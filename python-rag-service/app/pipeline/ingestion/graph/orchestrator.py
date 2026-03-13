import logging
from typing import List, Sequence

from app.schemas.graph import (
    GraphPayload,
    LawUnitNode,
    ReferenceEdge,
    PartOfEdge
)
from app.db.repository import LawUnitRepository
# Note: These extractors and repos will be built next
from app.pipeline.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.graph.extractors.semantic import SemanticExtractor
from app.pipeline.graph.resolution.clustering import ClusteringEngine
from app.db.graph import Neo4jRepository

logger = logging.getLogger(__name__)

class GraphOrchestrator:
    """
    Facade Pattern: Manages the Knowledge Graph ETL workflow[cite: 15].
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
        # Dependency Injection Pattern ensures loose coupling
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

        # 1. Fetch Source Data (PostgreSQL)
        units = []
        for uid in unit_ids:
            unit = await self.pg_repo.get(uid)
            if unit:
                units.append(unit)

        if not units:
            logger.warning("No valid LawUnits found for provided IDs.")
            return

        # Initialize the payload components
        law_unit_nodes: List[LawUnitNode] = []
        reference_edges: List[ReferenceEdge] = []
        part_of_edges: List[PartOfEdge] = []

        # 2. Process Layer 1: The Skeleton & Deterministic Edges [cite: 17]
        for unit in units:
            # Map Postgres entity to lightweight Graph DTO
            law_unit_nodes.append(LawUnitNode(id=unit.id, unit_type=unit.unit_type))

            # Reconstruct the structural hierarchy
            if unit.parent_id:
                part_of_edges.append(PartOfEdge(child_id=unit.id, parent_id=unit.parent_id))

            # Extract hard legal citations (e.g., Art. 5 references Art. 102) [cite: 16]
            refs = self.deterministic_extractor.extract_references(unit.id, unit.content)
            reference_edges.extend(refs)

        # 3. Process Layer 2 & 3: Semantic Extraction
        # GLiNER extracts raw strings and maps them to predefined Category Labels
        raw_entities = await self.semantic_extractor.extract_batch(units)

        # 4. Entity Resolution & Clustering
        # HDBSCAN dedupes and groups the raw strings into canonical ConceptNodes
        cluster_results = await self.clustering_engine.resolve_entities(raw_entities)

        # 5. Formulate the Final Payload (Data Transfer Object Pattern)
        payload = GraphPayload(
            law_units=law_unit_nodes,
            concepts=cluster_results.concepts,
            categories=cluster_results.categories,
            reference_edges=reference_edges,
            part_of_edges=part_of_edges,
            mentions_edges=cluster_results.mentions_edges,
            belongs_to_edges=cluster_results.belongs_to_edges
        )

        # 6. Load into Graph Database (Adapter Pattern)
        await self.graph_repo.upsert_payload(payload)
        logger.info(f"Successfully merged graph payload for {len(units)} units into Neo4j.")