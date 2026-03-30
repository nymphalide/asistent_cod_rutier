import logging
from typing import List, Dict, Any
from neo4j import AsyncDriver, AsyncTransaction

from app.schemas.graph import GraphPayload

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """
    Repository Pattern: Translates Pydantic domain models into Neo4j graph structures.
    Uses strict Dependency Injection for the driver to remain infrastructure-agnostic.
    """

    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def setup_constraints(self) -> None:
        """
        Ensures unique constraints exist for lightning-fast MERGE operations.
        Must be called once during application startup.
        """
        queries = [
            "CREATE CONSTRAINT law_unit_id IF NOT EXISTS FOR (n:LawUnit) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT external_law_id IF NOT EXISTS FOR (n:ExternalLawNode) REQUIRE n.id IS UNIQUE"
        ]

        async with self.driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.error(f"Failed to create constraint: {e}")
        logger.info("Neo4j Constraints verified.")

    async def upsert_payload(self, payload: GraphPayload) -> None:
        """
        Unit of Work Pattern: Executes the entire GraphPayload within a single transaction.
        If any part fails, the entire batch rolls back to prevent orphaned nodes.
        """
        async with self.driver.session() as session:
            try:
                await session.execute_write(self._execute_batch_upsert, payload)
            except Exception as e:
                logger.error(f"Neo4j Transaction Failed. Rolled back payload. Error: {e}")
                raise

    async def _execute_batch_upsert(self, tx: AsyncTransaction, payload: GraphPayload) -> None:
        """
        Executes the optimized UNWIND queries sequentially.
        Nodes MUST be created before Edges.
        """
        # 1. UPSERT NODES
        if payload.law_units:
            await self._upsert_law_units(tx, [u.model_dump(mode='json') for u in payload.law_units])

        if payload.concepts:
            await self._upsert_concepts(tx, [c.model_dump(mode='json') for c in payload.concepts])

        if payload.categories:
            await self._upsert_categories(tx, [c.model_dump(mode='json') for c in payload.categories])

        if payload.external_laws:
            await self._upsert_external_laws(tx, [n.model_dump(mode='json') for n in payload.external_laws])

        # 2. UPSERT EDGES
        if payload.reference_edges:
            await self._upsert_reference_edges(tx, [e.model_dump(mode='json') for e in payload.reference_edges])

        if payload.part_of_edges:
            await self._upsert_part_of_edges(tx, [e.model_dump(mode='json') for e in payload.part_of_edges])

        if payload.mentions_edges:
            await self._upsert_mentions_edges(tx, [e.model_dump(mode='json') for e in payload.mentions_edges])

        if payload.belongs_to_edges:
            await self._upsert_belongs_to_edges(tx, [e.model_dump(mode='json') for e in payload.belongs_to_edges])

        if payload.refers_to_external_edges:
            await self._upsert_refers_to_external_edges(tx, [e.model_dump(mode='json') for e in payload.refers_to_external_edges])

    # ==========================================
    # PRIVATE CYPHER QUERIES (Using UNWIND for Batching)
    # ==========================================

    async def _upsert_law_units(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MERGE (n:LawUnit {id: row.id})
        SET n.unit_type = row.unit_type
        """
        await tx.run(query, batch=data)

    async def _upsert_concepts(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MERGE (n:Concept {name: row.name})
        WITH n, row
        UNWIND (coalesce(n.surface_forms, []) + row.surface_forms) AS form
        WITH n, collect(DISTINCT form) AS unique_forms
        SET n.surface_forms = unique_forms
        """
        for row in data:
            row['surface_forms'] = list(row['surface_forms'])
        await tx.run(query, batch=data)

    async def _upsert_categories(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MERGE (n:Category {name: row.name})
        """
        await tx.run(query, batch=data)

    async def _upsert_external_laws(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MERGE (n:ExternalLawNode {id: row.id})
        SET n.name = row.name,
            n.law_type = row.law_type
        """
        await tx.run(query, batch=data)

    async def _upsert_reference_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MATCH (source:LawUnit {id: row.source_id})
        MATCH (target:LawUnit {id: row.target_id})
        MERGE (source)-[:REFERENCES]->(target)
        """
        await tx.run(query, batch=data)

    async def _upsert_part_of_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MATCH (child:LawUnit {id: row.child_id})
        MATCH (parent:LawUnit {id: row.parent_id})
        MERGE (child)-[:PART_OF]->(parent)
        """
        await tx.run(query, batch=data)

    async def _upsert_mentions_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MATCH (source:LawUnit {id: row.source_unit_id})
        MATCH (target:Concept {name: row.target_concept_name})
        MERGE (source)-[rel:MENTIONS]->(target)
        ON CREATE SET rel.extracted_texts = [row.extracted_text]
        ON MATCH SET rel.extracted_texts = CASE 
            WHEN NOT row.extracted_text IN rel.extracted_texts 
            THEN rel.extracted_texts + row.extracted_text 
            ELSE rel.extracted_texts 
        END
        """
        await tx.run(query, batch=data)

    async def _upsert_belongs_to_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MATCH (source:Concept {name: row.source_concept_name})
        MATCH (target:Category {name: row.target_category_name})
        MERGE (source)-[:BELONGS_TO]->(target)
        """
        await tx.run(query, batch=data)

    async def _upsert_refers_to_external_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        query = """
        UNWIND $batch AS row
        MATCH (source:LawUnit {id: row.source_unit_id})
        MATCH (target:ExternalLawNode {id: row.target_external_id})
        MERGE (source)-[:REFERS_TO_EXTERNAL]->(target)
        """
        await tx.run(query, batch=data)