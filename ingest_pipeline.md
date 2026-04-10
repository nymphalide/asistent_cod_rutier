backend python aplication

File: alembic\env.py
```py
import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context # type: ignore

# 2. Import your App Settings and Models
from app.core.config import settings
from app.db.session import Base
from app.db.models import LawUnit # Explicitly import models to register them!

# --- CUSTOM CONFIGURATION END ---

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 3. Set the MetaData target (This tells Alembic what the DB "Should" look like)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    # Overwrite the alembic.ini url with the one from our environment variables
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # 4. Overwrite config URL with Pydantic settings
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.DATABASE_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

File: app\__init__.py
```py

```

File: app\api\dependencies.py
```py
# app/api/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session per request.
    Implements the Unit of Work pattern: automatically commits on success,
    and rolls back on any exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            # Hand the session to the FastAPI endpoint
            yield session

            # If the endpoint finishes without errors, commit the transaction
            await session.commit()

        except Exception:
            # If the endpoint crashes for ANY reason, undo everything
            await session.rollback()
            raise
```

File: app\core\ai_registry.py
```py
from typing import Dict, Any


class ModelRegistry:
    """
    Centralized registry for AI model hyperparameters.
    Separates behavioral AI logic from infrastructure environment variables.
    """

    @staticmethod
    def get_enricher_chat_config() -> Dict[str, Any]:
        """Configuration for the QuOTE strategy question generation."""
        return {
            "model": "llama3.1:8b",
            "format": "json",
            "options": {
                "temperature": 0.3,  # Slight variance to get distinct questions
                "top_p": 0.9,
            }
        }

    @staticmethod
    def get_embedding_model() -> str:
        """The designated embedding model for dense vectors."""
        return "nomic-embed-text"

    @staticmethod
    def get_embedding_dimensions() -> int:
        """Required by Qdrant to initialize the vector index."""
        return 768  # nomic-embed-text outputs 768 dimensions

    @staticmethod
    def get_ner_config() -> Dict[str, Any]:
        """Configuration for the Zero-Shot NER extraction."""
        return {
            "model": "urchade/gliner_multi-v2.1",
            "batch_size": 8
        }

    @staticmethod
    def get_clustering_config() -> Dict[str, Any]:
        """Hyperparameters for HDBSCAN and entity resolution."""
        return {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "cosine"
        }
```

File: app\core\config.py
```py
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, validator
from typing import Optional


class Settings(BaseSettings):
    """
    Configuration for the Python RAG microservice. [cite: 112]
    This class VALIDATES the environment; it does not set the truth.
    """
    PROJECT_NAME: str = "Optimised Romanian Traffic Code RAG"
    # --- File Paths ---
    RAW_DATA_DIR: str = "data/raw_text"

    # --- Databases (No defaults here forces them to be in .env) ---
    DATABASE_URL: str = Field(validation_alias="PYTHON_DATABASE_URL")
    QDRANT_HOST: str  # Required
    QDRANT_PORT: int  # Required
    QDRANT_COLLECTION: str = "traffic_code_vectors"

    # Graph Store is optional in the plan [cite: 15]
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None


    # --- Local AI Settings (Ollama) ---
    # WSL2 usually resolves localhost fine, but if Docker bridge fails later, use host.docker.internal
    OLLAMA_HOST: str = "http://localhost:11434"
    MAX_CONCURRENT_ENRICHMENT_TASKS: int = 5

    # --- Hardware ---
    USE_CUDA: bool = True

    @property
    def DEVICE(self) -> str:
        """Dynamically selects the GPU (RTX 4060) if available."""
        if self.USE_CUDA and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # Crucial: This tells Pydantic WHERE the truth is.
    model_config = SettingsConfigDict(
        # Pydantic reads left-to-right.
        # It loads .env first, then overwrites anything it finds in .env.local
        env_file=("../.env", "../.env.local"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
```

File: app\core\custom_types.py
```py
# app/core/custom_types.py
from enum import Enum

class UnitType(str, Enum):
    CHAPTER = 'chapter'         # Top-level division (e.g., "CAPITOLUL I: DISPOZITII GENERALE")
    SECTION = 'section'         # Sub-division of a chapter (e.g., "SECŢIUNEA 1")
    ARTICLE = 'article'         # The main legal node (e.g., "Art. 5", "Art. 11.2")
    PROLOGUE = 'prologue'       # Unnumbered intro text directly under an Article, before paragraphs start (e.g., the first sentence in Art. 6)
    PARAGRAPH = 'paragraph'     # Numbered paragraphs using parentheses (e.g., "(1)", "(1.1)", "(2)")
    LETTER_ITEM = 'letter_item' # Lower-level lists using letters (e.g., "a)", "b)", "c)") inside paragraphs or annexes
    NUMBERED_ITEM = 'num_item'  # Numbered lists without parentheses (e.g., definitions like "1.", "35.1." or Annex items)
```

File: app\db\__init__.py
```py
# Expose key components to the rest of the app
from .session import Base, AsyncSessionLocal, engine
from .models import LawUnit
from .repository import LawUnitRepository
```

File: app\db\graph.py
```py
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
```

File: app\db\models.py
```py
from typing import Optional, List, Any, Dict
from sqlalchemy import String, Text, ForeignKey, Computed
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, ENUM

from app.db.session import Base
# We will define the Enum in schemas next, but we reference it here for the DB type.
# Ensure your schema file has the UnitType enum defined as verified in the next step.
from app.core.custom_types import UnitType
from app.core.config import settings


class LawUnit(Base):
    """
    SQLAlchemy model representing the 'law_units' table.
    Stores the graph nodes (Articles, Paragraphs) and their vector embeddings.
    """
    __tablename__ = "law_units"

    # --- Primary Keys & Identifiers ---
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)

    # --- Hierarchy & Graph Relationships ---
    # Structural Parent (e.g., Art 5 -> Chapter II)
    parent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("law_units.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # --- Content ---
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # We use the Python Enum 'UnitType' to strictly type this column in Postgres
    unit_type: Mapped[UnitType] = mapped_column(
        # FIX: Ensure the enum inherits schema metadata so Alembic can track it
        ENUM(UnitType, name="unit_type_enum", create_type=True, metadata=Base.metadata),
        nullable=False
    )

    # --- Metadata ---
    # Stores flexible data like {"chapter_title": "...", "breadcrumbs": [...]}
    # We map it to the SQL column 'meta_info' (requires a migration!)
    meta_info: Mapped[Dict[str, Any]] = mapped_column("meta_info", JSONB, server_default='{}')


    # --- ORM Relationships (Optional helper for Python navigation) ---
    # Allows accessing children via parent.children
    children: Mapped[List["LawUnit"]] = relationship(
        "LawUnit",
        back_populates="parent",
        foreign_keys=[parent_id],
        cascade="all, delete-orphan"
    )

    parent: Mapped[Optional["LawUnit"]] = relationship(
        "LawUnit",
        remote_side=[id],
        foreign_keys=[parent_id],
        back_populates="children"
    )

    def __repr__(self):
        return f"<LawUnit(id='{self.id}', type='{self.unit_type}', parent='{self.parent_id}')>"
```

File: app\db\repository.py
```py
from typing import List, Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import case

from app.db.models import LawUnit
from app.schemas.law_unit import LawUnitCreate

class LawUnitRepository:
    """
    Repository for handling LawUnit (Traffic Code) database operations.
    Acts as the 'Librarian', abstracting SQL away from the business logic.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, unit_in: LawUnitCreate) -> LawUnit:
        """
        Creates a single LawUnit record by converting the Pydantic
        schema into a SQLAlchemy model instance.
        """
        values = unit_in.model_dump(by_alias=False)
        db_obj = LawUnit(**values)

        self.db.add(db_obj)
        # We use flush() instead of commit() to respect the Unit of Work pattern.
        # This pushes the query to the DB to get generated fields without finalizing the transaction.
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def get(self, unit_id: str) -> Optional[LawUnit]:
        """
        Retrieves a LawUnit by its ID using async execute.
        """
        stmt = select(LawUnit).where(LawUnit.id == unit_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_children(self, parent_id: str) -> Sequence[LawUnit]:
        """
        Retrieves all direct children of a specific unit using async execute.
        """
        stmt = select(LawUnit).where(LawUnit.parent_id == parent_id)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def bulk_upsert(self, units_in: List[LawUnitCreate]) -> int:
        if not units_in:
            return 0

        #extracts only the lawcreateunitcreate fields from the enriched unit for the postgres upsert
        postgres_fields = LawUnitCreate.model_fields.keys()

        values = [
            obj.model_dump(by_alias=False, include=postgres_fields)
            for obj in units_in
        ]

        stmt = insert(LawUnit).values(values)

        # Grab every column from the schema EXCEPT id to update on conflict
        update_dict = {
            col.name: col
            for col in stmt.excluded
            if col.name != "id"
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=[LawUnit.id],
            set_=update_dict
        )

        await self.db.execute(stmt)
        return len(values)
```

File: app\db\session.py
```py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

# create_async_engine replaces create_engine
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)

# async_sessionmaker replaces sessionmaker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()
```

File: app\db\vector.py
```py
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
```

File: app\pipeline\__init__.py
```py

```

File: app\pipeline\ingestion\__init__.py
```py
import os
import glob
import logging
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.law_unit import LawUnitCreate, LawUnitEnriched
from app.pipeline.ingestion.parser import TrafficCodeParser
from app.pipeline.ingestion.enricher import EnricherService
from app.db.repository import LawUnitRepository
from app.db.vector import QdrantRepository
from app.db.graph import Neo4jRepository
from app.pipeline.ingestion.graph.orchestrator import GraphOrchestrator
from app.pipeline.ingestion.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from app.pipeline.ingestion.graph.clustering import ClusteringEngine

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Facade Pattern for orchestrating the RAG ETL process.
    Uses strict Dependency Injection for database connections.
    """

    def __init__(self, pg_session: AsyncSession, qdrant_repo: QdrantRepository, graph_repo: Neo4jRepository):
        self.pg_session = pg_session
        self.pg_repo = LawUnitRepository(self.pg_session)
        self.qdrant_repo = qdrant_repo
        self.enricher = EnricherService()
        self.graph_orchestrator = GraphOrchestrator(
            pg_repo=self.pg_repo,
            graph_repo=graph_repo,
            deterministic_extractor=DeterministicExtractor(),
            semantic_extractor=SemanticExtractor(),
            clustering_engine=ClusteringEngine()
        )

    async def process_directory(self, raw_data_dir: str) -> List[LawUnitEnriched]:
        logger.info(f"--- Starting Ingestion from: {raw_data_dir} ---")

        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

        # 1. Gather Files
        file_paths = glob.glob(os.path.join(raw_data_dir, "*.txt"))
        file_paths.sort()

        if not file_paths:
            logger.warning("No .txt files found.")
            return []

        all_units: List[LawUnitCreate] = []

        # 2. Parse Files
        logger.info(f"Parsing {len(file_paths)} files...")
        for file_path in file_paths:
            logger.info(f" -> Processing: {os.path.basename(file_path)}")
            parser = TrafficCodeParser(file_path)
            file_units = parser.parse()
            all_units.extend(file_units)

        logger.info(f"Parsing complete. Extracted {len(all_units)} units.")

        # 3. Enrich (Embeddings & Hypothetical Questions)
        logger.info("Enriching data with Embeddings and Questions...")
        enriched_units = await self.enricher.enrich_batch(all_units)

        # 4. Save to Database (Saga Pattern / Distributed Transaction)
        logger.info("Writing to Databases...")

        # We do not use 'try...except' to commit/rollback here.
        # The Unit of Work pattern dictates that the caller managing the session handles the transaction.
        count = await self.pg_repo.bulk_upsert(enriched_units)
        await self.qdrant_repo.bulk_upsert(enriched_units)

        logger.info(f"SUCCESS: Mapped vectors to Qdrant and prepped {count} units for Postgres commit.")

        # 5. Extract and Load Knowledge Graph
        logger.info("Extracting Knowledge Graph boundaries and semantics...")
        unit_ids = [u.id for u in enriched_units]

        # Now this will actually work because it's a real instantiated object
        await self.graph_orchestrator.process_batch(unit_ids)

        return enriched_units
```

File: app\pipeline\ingestion\enricher.py
```py
import json
import logging
import asyncio
from typing import List, Optional
from ollama import AsyncClient

from app.schemas.law_unit import LawUnitCreate, LawUnitEnriched
from app.core.custom_types import UnitType
from app.core.config import settings
from app.core.ai_registry import ModelRegistry

logger = logging.getLogger(__name__)


class EnricherService:
    """
    Service Layer responsible for AI enrichment (Embeddings & QuOTE Strategy).
    """

    def __init__(self):
        self.client = AsyncClient(host=settings.OLLAMA_HOST)
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_ENRICHMENT_TASKS)

    async def _generate_questions(self, content: str) -> List[str]:
        prompt = (
            "You are a Romanian legal expert. Read the following traffic code text "
            "and generate exactly 3 hypothetical questions that a driver might ask, "
            "which are directly answered by this text. "
            "Respond ONLY in valid JSON format using the following schema: "
            '{"questions": ["question 1", "question 2", "question 3"]}\n\n'
            f"Text: {content}"
        )

        config = ModelRegistry.get_enricher_chat_config()

        try:
            response = await self.client.chat(
                messages=[{'role': 'user', 'content': prompt}],
                **config
            )

            result = json.loads(response['message']['content'])
            return result.get("questions", [])[:3]

        except Exception as e:
            logger.error(f"LLM Question Generation failed: {e}")
            return []

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            return None

        try:
            response = await self.client.embeddings(
                model=ModelRegistry.get_embedding_model(),
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def enrich_unit(self, unit: LawUnitCreate) -> LawUnitEnriched:
        async with self.semaphore:

            # 1. Skip QuOTE for Structural Headers (Chapters/Sections don't answer questions)
            if unit.unit_type in [UnitType.CHAPTER, UnitType.SECTION]:
                content_vector = await self._generate_embedding(unit.content)
                return LawUnitEnriched(
                    **unit.model_dump(by_alias=True),
                    hypothetical_questions=[],
                    content_vector=content_vector,
                    question_vectors=[]
                )

            # 2. Generate QuOTE Questions
            questions = await self._generate_questions(unit.content)

            # 3. Parallel Vector Generation (Content + N Questions)
            tasks = [self._generate_embedding(unit.content)]
            for q in questions:
                tasks.append(self._generate_embedding(q))

            embeddings = await asyncio.gather(*tasks)

            content_vector = embeddings[0]
            question_vectors = [vec for vec in embeddings[1:] if vec is not None]

            # 4. Return the Enriched Contract
            return LawUnitEnriched(
                **unit.model_dump(by_alias=True),
                hypothetical_questions=questions,
                content_vector=content_vector,
                question_vectors=question_vectors
            )

    async def enrich_batch(self, units: List[LawUnitCreate]) -> List[LawUnitEnriched]:
        logger.info(f"Starting GPU enrichment for {len(units)} units...")

        tasks = [self.enrich_unit(unit) for unit in units]
        enriched_units = await asyncio.gather(*tasks)

        logger.info("Batch enrichment complete.")
        return enriched_units
```

File: app\pipeline\ingestion\graph\__init__.py
```py

```

File: app\pipeline\ingestion\graph\clustering.py
```py
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import HDBSCAN
from collections import Counter
from pydantic import BaseModel, Field

from app.schemas.graph import (
    RawEntity,
    ConceptNode,
    CategoryNode,
    MentionsEdge,
    BelongsToEdge
)
from app.core.ai_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ClusteringResult(BaseModel):
    """DTO ensuring the Orchestrator receives the exact attributes it expects."""
    concepts: List[ConceptNode] = Field(default_factory=list)
    categories: List[CategoryNode] = Field(default_factory=list)
    mentions_edges: List[MentionsEdge] = Field(default_factory=list)
    belongs_to_edges: List[BelongsToEdge] = Field(default_factory=list)


class ClusteringEngine:
    """
    Implements Entity Resolution.
    Matches the Orchestrator's expectation for 'resolve_entities' method.
    """

    def __init__(self, embedder_service):
        # 1. Infrastructure: Pull the host from Pydantic Settings
        self.sync_client = Client(host=settings.OLLAMA_HOST)

        # 2. Behavior: Pull hyperparameters from the Registry
        self.config = ModelRegistry.get_clustering_config()
        self.embedding_model = ModelRegistry.get_embedding_model()

    async def resolve_entities(self, raw_entities: List[RawEntity]) -> ClusteringResult:
        """Entry point as called by GraphOrchestrator."""
        if not raw_entities:
            return ClusteringResult()

        return await asyncio.to_thread(self._sync_resolve, raw_entities)

    def _sync_resolve(self, raw_entities: List[RawEntity]) -> ClusteringResult:
        # 1. Deduplicate for embedding efficiency
        unique_forms = list(set(e.surface_form for e in raw_entities))
        embeddings = self.embedder.embed_documents(unique_forms)
        matrix = np.array(embeddings)

        # 2. HDBSCAN Clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.config["min_cluster_size"],
            min_samples=self.config["min_samples"],
            metric=self.config["metric"]
        )
        labels = clusterer.fit_predict(matrix)

        # 3. Formulate Nodes and Edges
        result = ClusteringResult()
        clusters: Dict[int, List[str]] = {}
        category_set = set()

        for idx, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(label, []).append(unique_forms[idx])

        for label, forms in clusters.items():
            canonical_name = min(forms, key=len)
            category_name = self._get_dominant_category(canonical_name, raw_entities)
            concept_id = f"concept_{canonical_name.replace(' ', '_')}"

            # Create Concept
            result.concepts.append(ConceptNode(
                id=concept_id,
                name=canonical_name,
                surface_forms=list(set(forms))
            ))

            # Track unique categories to create CategoryNodes
            category_set.add(category_name)

            # Create Layer 3 Edge (Concept -> Category)
            result.belongs_to_edges.append(BelongsToEdge(
                source_concept_id=concept_id,
                target_category_name=category_name
            ))

            # Create Layer 2 Edges (LawUnit -> Concept)
            for entity in raw_entities:
                if entity.surface_form in forms:
                    result.mentions_edges.append(MentionsEdge(
                        source_unit_id=entity.source_unit_id,
                        target_concept_id=concept_id
                    ))

        # Finalize Category Nodes
        result.categories = [CategoryNode(name=cat) for cat in category_set]

        return result

    def _get_dominant_category(self, canonical_name: str, raw_entities: List[RawEntity]) -> str:
        relevant = [e.category_label for e in raw_entities if e.surface_form == canonical_name]
        return Counter(relevant).most_common(1)[0][0] if relevant else "General"
```

File: app\pipeline\ingestion\graph\extractors\__init__.py
```py

```

File: app\pipeline\ingestion\graph\extractors\deterministic.py
```py
import re
import logging
from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.graph import ReferenceEdge, ExternalLawNode, RefersToExternalEdge

logger = logging.getLogger(__name__)


# ==========================================
# DTO PATTERN: EXTRACTOR OUTPUT
# ==========================================

class DeterministicResult(BaseModel):
    """
    Packages the Layer 1 graph boundaries for the Orchestrator.
    Prevents the Orchestrator from needing to know how the regex works.
    """
    internal_edges: list[ReferenceEdge] = Field(default_factory=list)
    external_nodes: list[ExternalLawNode] = Field(default_factory=list)
    external_edges: list[RefersToExternalEdge] = Field(default_factory=list)


# ==========================================
# STRATEGY PATTERN: REGEX PARSING
# ==========================================

class DeterministicExtractor:
    """
    Parses legal text to deterministically build the Layer 1
    structural Knowledge Graph skeleton.
    """

    # 1. External Legislation Regex
    # Captures: "Legea nr. 286/2009", "Ordonanța de urgență nr. 195/2002", "Regulamentul (UE) nr. 168/2013"
    REGEX_EXTERNAL = re.compile(
        r"(?P<type>Legea|Ordonan[tț][aă](?:\s+Guvernului|\s+de\s+urgen[tț][aă])?|Regulamentul\s*\(UE\))\s+nr\.\s*(?P<num>\d+/\d{4})",
        re.IGNORECASE
    )

    # 2. Internal Cross-References Regex (Unified with Named Groups)
    # The order of these branches matters; it evaluates the most complex first.
    REGEX_INTERNAL = re.compile(
        r"(?P<abs_art_range>art\.\s*(?P<r_start>\d+(?:\.\d+)?)\s*-\s*(?P<r_end>\d+(?:\.\d+)?))|"
        r"(?P<abs_art_alin>art\.\s*(?P<aa_art>\d+(?:\.\d+)?)\s*alin\.\s*\((?P<aa_alin>[a-z0-9.]+)\))|"
        r"(?P<abs_art>art\.\s*(?P<a_art>\d+(?:\.\d+)?))|"
        r"(?P<rel_alin>alin\.\s*\((?P<r_alin>[a-z0-9.]+)\))",
        re.IGNORECASE
    )

    def extract_references(self, source_id: str, content: str) -> DeterministicResult:
        """Main execution method for deterministic parsing."""
        result = DeterministicResult()

        self._parse_external(source_id, content, result)
        self._parse_internal(source_id, content, result)

        return result

    def _parse_external(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds boundaries triggering external routing strategies."""
        for match in self.REGEX_EXTERNAL.finditer(content):
            law_type = match.group('type').strip().title()
            num = match.group('num').strip()

            # Normalize ID: "Legea nr. 286/2009" -> "legea_286_2009"
            ext_id = f"{law_type.split()[0].lower()}_{num.replace('/', '_')}"
            ext_name = f"{law_type} nr. {num}"

            result.external_nodes.append(
                ExternalLawNode(id=ext_id, name=ext_name, law_type=law_type)
            )
            result.external_edges.append(
                RefersToExternalEdge(source_unit_id=source_id, target_external_id=ext_id)
            )

    def _parse_internal(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds internal nodes to build standard traversal paths."""
        base_article = self._get_base_article(source_id)

        for match in self.REGEX_INTERNAL.finditer(content):
            # Clean the dictionary of None values to enable exact structural pattern matching
            valid_groups = {k: v for k, v in match.groupdict().items() if v is not None}

            match valid_groups:
                # Format: "art. 11.2 - 11.4"
                case {"abs_art_range": _, "r_start": start, "r_end": end}:
                    start_id = f"art_{start.replace('.', '_')}"
                    end_id = f"art_{end.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=start_id))
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=end_id))

                # Format: "art. 13 alin. (3)"
                case {"abs_art_alin": _, "aa_art": art, "aa_alin": alin}:
                    target_id = f"art_{art.replace('.', '_')}_alin_{alin.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Format: "art. 102"
                case {"abs_art": _, "a_art": art}:
                    target_id = f"art_{art.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Format: "alin. (3)" -> Requires Relative Context
                case {"rel_alin": _, "r_alin": alin}:
                    if base_article:
                        target_id = f"{base_article}_alin_{alin.replace('.', '_')}"
                        result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Fallback for safe degradation
                case _:
                    logger.debug(f"Regex matched but fell through structural matching on ID {source_id}")

    def _get_base_article(self, source_id: str) -> Optional[str]:
        """
        Extracts the parent article ID to resolve relative references.
        Example: 'art_102_alin_3_lit_a' -> 'art_102'
        """
        match = re.match(r"^(art_\d+(?:_\d+)?)", source_id)
        if match:
            return match.group(1)
        return None
```

File: app\pipeline\ingestion\graph\extractors\semantic.py
```py
import os
import json
import logging
import asyncio
from typing import List
from gliner import GLiNER

from app.schemas.graph import RawEntity
from app.db.models import LawUnit
from app.core.config import settings
from app.core.ai_registry import ModelRegistry

logger = logging.getLogger(__name__)


class SemanticExtractor:
    def __init__(self, ontology_path: str = "app/pipeline/ingestion/graph/ontology.json"):
        self.ontology_path = ontology_path
        self.labels = self._load_ontology()

        # Pulling behavioral config from Registry, infrastructure from Settings
        config = ModelRegistry.get_ner_config()
        self.model_name = config["model"]
        self.batch_size = config["batch_size"]
        self.device = settings.DEVICE

        logger.info(f"Initializing {self.model_name} on {self.device}...")
        try:
            self.model = GLiNER.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _load_ontology(self) -> List[str]:
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f).get("labels", [])

    def _sync_predict_batch(self, units: List[LawUnit]) -> List[RawEntity]:
        texts = [unit.content for unit in units]
        raw_entities: List[RawEntity] = []

        batch_results = self.model.batch_predict_entities(
            texts,
            self.labels,
            batch_size=self.batch_size
        )

        for unit, entities in zip(units, batch_results):
            for entity in entities:
                clean_text = entity["text"].strip().lower()
                if clean_text:
                    raw_entities.append(
                        RawEntity(
                            surface_form=clean_text,
                            category_label=entity["label"],
                            source_unit_id=unit.id
                        )
                    )
        return raw_entities

    async def extract_batch(self, units: List[LawUnit]) -> List[RawEntity]:
        return await asyncio.to_thread(self._sync_predict_batch, units)
```

File: app\pipeline\ingestion\graph\orchestrator.py
```py
import logging
from typing import List

from app.schemas.graph import (
    GraphPayload,
    LawUnitNode,
    PartOfEdge
)
from app.db.repository import LawUnitRepository
from app.db.graph import Neo4jRepository
from app.pipeline.ingestion.graph.extractors.deterministic import DeterministicExtractor
from app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from app.pipeline.ingestion.graph.clustering import ClusteringEngine

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
```

File: app\pipeline\ingestion\parser.py
```py
import re
from typing import List, Optional, Dict, Any, Tuple
from app.schemas.law_unit import LawUnitCreate
from app.core.custom_types import UnitType


class TrafficCodeParser:
    REGEX_CHAPTER = re.compile(r"^(CAPITOLUL|TITLUL)\s+([IVXLCDM]+)(?::\s*(.*))?", re.IGNORECASE)
    REGEX_SECTION = re.compile(r"^SEC[TȚ]IUNEA\s+(?:nr\.\s*)?(\d+|a\s+[a-z0-9-]+-a)(?:\s*(.*))?", re.IGNORECASE)
    REGEX_ART = re.compile(r"^Art\.\s*(\d+(?:\.\d+)?)\.?\s*(.*)", re.IGNORECASE)
    REGEX_PARA = re.compile(r"^\((\d+(?:\.\d+)?)\)\s*(.*)")
    REGEX_LETTER = re.compile(r"^([a-z])\)\s*(.*)")
    REGEX_NUM_ITEM = re.compile(r"^(\d+(?:\.\d+)?)\.\s*(.*?)(?:\s*-\s*(.*))?$")

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results: List[LawUnitCreate] = []

        self.active_chapter: Optional[str] = None
        self.active_section: Optional[str] = None
        self.active_article: Optional[str] = None
        self.active_paragraph: Optional[str] = None
        self.active_list_parent: Optional[str] = None

        self.current_buffer: List[str] = []
        self.pending_meta: Optional[Dict[str, Any]] = None

    def _read_file_safely(self, file_path: str) -> List[str]:
        encodings = ['utf-8-sig', 'iso-8859-2', 'cp1252', 'utf-8']

        # Mapping legacy cedillas to correct comma-below diacritics
        diacritic_map = str.maketrans({'ş': 'ș', 'ţ': 'ț', 'Ş': 'Ș', 'Ţ': 'Ț'})

        for enc in encodings:
            try:
                # Adding newline=None forces Python to translate all \r\n to \n automatically
                with open(file_path, 'r', encoding=enc, newline=None) as f:
                    lines = []
                    for line in f.readlines():
                        # strip('\r\n ') guarantees no Windows ghosts survive
                        clean_line = line.strip('\r\n ')
                        if clean_line:
                            # Normalize text to bulletproof downstream deterministic extraction
                            lines.append(clean_line.translate(diacritic_map))
                    return lines
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Encoding failed for {file_path}")

    def _flush_buffer(self):
        if self.pending_meta and self.current_buffer:
            content = " ".join(self.current_buffer)
            self.results.append(LawUnitCreate(content=content, **self.pending_meta))

        self.current_buffer = []
        self.pending_meta = None

    def _tokenize(self, line: str) -> Tuple[str, Any]:
        """The Lexer: Returns a structural tuple (Token_Type, Match_Data)"""
        if match := self.REGEX_CHAPTER.match(line): return ('CHAPTER', match)
        if match := self.REGEX_SECTION.match(line): return ('SECTION', match) # <-- ADDED
        if match := self.REGEX_ART.match(line): return ('ARTICLE', match)
        if match := self.REGEX_PARA.match(line): return ('PARAGRAPH', match)
        if match := self.REGEX_LETTER.match(line): return ('LETTER', match)
        if match := self.REGEX_NUM_ITEM.match(line): return ('NUMBERED', match)
        return ('TEXT', line)

    def parse(self) -> List[LawUnitCreate]:
        lines = self._read_file_safely(self.file_path)

        for line in lines:
            token_type, match_data = self._tokenize(line)

            # Universal Pre-processing: Flush if a NEW structural unit begins
            if token_type != 'TEXT':
                self._flush_buffer()

            match token_type:
                case 'CHAPTER':
                    num, title = match_data.group(2), match_data.group(3) or line
                    self.active_chapter = f"cap_{num}"
                    self.active_section = self.active_article = self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_chapter, "parent_id": None, "unit_type": UnitType.CHAPTER, "metadata": {"title": title}}

                case 'SECTION':
                    sec_num = match_data.group(1).replace(' ', '_')
                    title = match_data.group(2) or line
                    unit_id = f"{self.active_chapter}_sec_{sec_num}" if self.active_chapter else f"sec_{sec_num}"
                    self.active_section = unit_id
                    self.active_article = self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_section, "parent_id": self.active_chapter, "unit_type": UnitType.SECTION, "metadata": {"title": title, "number": sec_num}}

                case 'ARTICLE':
                    art_num = match_data.group(1)
                    self.active_article = f"art_{art_num.replace('.', '_')}"
                    self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_article, "parent_id": self.active_section or self.active_chapter, "unit_type": UnitType.ARTICLE, "metadata": {"number": art_num}}

                case 'PARAGRAPH':
                    para_num = match_data.group(1)
                    self.active_paragraph = f"{self.active_article}_alin_{para_num.replace('.', '_')}"
                    self.active_list_parent = self.active_paragraph
                    self.pending_meta = {"id": self.active_paragraph, "parent_id": self.active_article, "unit_type": UnitType.PARAGRAPH, "metadata": {"number": para_num}}

                case 'LETTER':
                    letter = match_data.group(1)
                    unit_id = f"{self.active_list_parent or self.active_article}_lit_{letter}"
                    self.pending_meta = {"id": unit_id, "parent_id": self.active_list_parent or self.active_article, "unit_type": UnitType.LETTER_ITEM, "metadata": {"letter": letter}}

                case 'NUMBERED':
                    num = match_data.group(1)
                    unit_id = f"{self.active_article or 'anexa'}_pct_{num.replace('.', '_')}"
                    self.pending_meta = {"id": unit_id, "parent_id": self.active_list_parent or self.active_article, "unit_type": UnitType.NUMBERED_ITEM, "metadata": {"number": num}}

                case 'TEXT':
                    # Edge case: Catching unnumbered prologue text before paragraphs start
                    if not self.current_buffer and self.active_article and not self.active_paragraph:
                        self.pending_meta = {"id": f"{self.active_article}_prologue", "parent_id": self.active_article, "unit_type": UnitType.PROLOGUE, "metadata": {}}

            # Universal Post-processing: Always append the current line to the active buffer
            self.current_buffer.append(line)

        self._flush_buffer()
        return self.results
```

File: app\schemas\graph.py
```py
from typing import List, Set
from pydantic import BaseModel, Field
from app.core.custom_types import UnitType


# ==========================================
# INTERMEDIATE PIPELINE DTOs
# ==========================================

class RawEntity(BaseModel):
    """
    Output from the GLiNER Semantic Extractor before clustering.
    Captures the exact string and its predicted category.
    """
    surface_form: str = Field(..., description="Exact string found in text (e.g., 'amenzii')")
    category_label: str = Field(..., description="Ontology label assigned by GLiNER (e.g., 'Sancțiuni')")
    source_unit_id: str = Field(..., description="The LawUnit ID where this was found")


# ==========================================
# GRAPH NODES (ENTITIES)
# ==========================================

class LawUnitNode(BaseModel):
    """
    Layer 1: The Skeleton.
    A lightweight routing node mapped 1:1 with Postgres.
    Excludes the heavy 'content' field to optimize Graph RAM.
    """
    id: str = Field(..., description="Canonical ID from Postgres (e.g., 'art_5')")
    unit_type: UnitType = Field(..., description="Enum defining the hierarchy level")


class ConceptNode(BaseModel):
    """
    Layer 2: The Canonical Concept post-HDBSCAN clustering.
    """
    name: str = Field(..., description="The mathematical centroid name")
    surface_forms: Set[str] = Field(
        default_factory=set,
        description="Deduplicated array of all exact string variations"
    )


class CategoryNode(BaseModel):
    """
    Layer 3: The Topology Ontology Label.
    Implements the Registry Pattern for static labels.
    """
    name: str = Field(..., description="The static ontology label (e.g., 'Infracțiuni')")


class ExternalLawNode(BaseModel):
    """
    Layer 1 Boundary: Represents legislation outside the Traffic Code.
    Acts as a deterministic trigger for the DSPy Web Search routing strategy.
    """
    id: str = Field(..., description="Normalized canonical ID (e.g., 'legea_286_2009')")
    name: str = Field(..., description="Full text match (e.g., 'Legea nr. 286/2009')")
    law_type: str = Field(..., description="Classification (e.g., 'Lege', 'Ordonanță', 'Regulament UE')")


# ==========================================
# GRAPH EDGES (RELATIONSHIPS)
# ==========================================

class ReferenceEdge(BaseModel):
    """Layer 1: Deterministic cross-reference ([:REFERENCES])."""
    source_id: str = Field(..., description="ID of the LawUnit making the citation")
    target_id: str = Field(..., description="ID of the referenced LawUnit")


class PartOfEdge(BaseModel):
    """Layer 1 Hierarchy: Structural parent-child relationship ([:PART_OF])."""
    child_id: str = Field(..., description="ID of the subordinate LawUnit (e.g., Paragraph)")
    parent_id: str = Field(..., description="ID of the parent LawUnit (e.g., Article)")


class MentionsEdge(BaseModel):
    """Layer 2: Semantic bridge from text to concept ([:MENTIONS])."""
    source_unit_id: str = Field(..., description="ID of the LawUnit")
    target_concept_name: str = Field(..., description="Name of the Canonical Concept")
    extracted_text: str = Field(..., description="The specific surface form that triggered this connection")


class BelongsToEdge(BaseModel):
    """Layer 3: Concept classification ([:BELONGS_TO])."""
    source_concept_name: str = Field(..., description="Name of the Canonical Concept")
    target_category_name: str = Field(..., description="Name of the Category")


class RefersToExternalEdge(BaseModel):
    """
    Layer 1 Boundary: LawUnit -> ExternalLawNode connection ([:REFERS_TO_EXTERNAL]).
    """
    source_unit_id: str = Field(..., description="ID of the internal LawUnit making the citation")
    target_external_id: str = Field(..., description="Normalized ID of the ExternalLawNode")


# ==========================================
# MASTER FACADE PAYLOAD
# ==========================================

class GraphPayload(BaseModel):
    """
    The final validated package the Orchestrator hands to the Neo4j Adapter.
    Enforces a strict contract between the ML pipeline and the database layer.
    """
    # Nodes
    law_units: List[LawUnitNode] = Field(default_factory=list)
    concepts: List[ConceptNode] = Field(default_factory=list)
    categories: List[CategoryNode] = Field(default_factory=list)
    external_laws: List[ExternalLawNode] = Field(default_factory=list)

    # Edges
    reference_edges: List[ReferenceEdge] = Field(default_factory=list)
    part_of_edges: List[PartOfEdge] = Field(default_factory=list)
    mentions_edges: List[MentionsEdge] = Field(default_factory=list)
    belongs_to_edges: List[BelongsToEdge] = Field(default_factory=list)
    refers_to_external_edges: List[RefersToExternalEdge] = Field(default_factory=list)
```

File: app\schemas\law_unit.py
```py
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field
from app.core.custom_types import UnitType

# 1. The Base Model
# Properties shared by all versions of the object
class LawUnitBase(BaseModel):
    id: str = Field(..., description="Canonical ID (e.g., 'art_102_alin_2')")
    content: str = Field(..., description="The actual legal text")
    unit_type: UnitType

    # Metadata is a flexible dictionary for things like breadcrumbs or chapter titles
    # FIX: Use 'alias' so the API receives "metadata", but Python sees "meta_info"
    meta_info: Dict[str, Any] = Field(default_factory=dict, alias="metadata")

    # FIX: Allow population by name or alias
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# 2. Creation DTO (Input)
# Used by the Ingestion Pipeline to WRITE data (for postgres and other uses)
class LawUnitCreate(LawUnitBase):
    parent_id: Optional[str] = None

#3. Enrichement DTO
#Used for communication with the Qdrant database
class LawUnitEnriched(LawUnitCreate):
    """
    DTO carrying the AI-enriched data destined for Qdrant.
    """
    hypothetical_questions: List[str] = Field(
        default_factory=list,
        description="Distinct questions generated by the Ollama LLM"
    )
    content_vector: Optional[List[float]] = Field(
        default=None,
        description="Dense embedding of the raw legal content"
    )
    question_vectors: Optional[List[List[float]]] = Field(
        default=None,
        description="List of dense embeddings, mapping 1:1 to hypothetical_questions"
    )

# 4. Response DTO (Output)
# Used by the API to READ data (in the future will be sent to the spring application)
class LawUnitResponse(LawUnitBase):
    parent_id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)
```

File: main.py
```py
import torch
from fastapi import FastAPI
from app.core.config import settings
from app.db import engine, Base
from app.db import models
app = FastAPI(title=settings.PROJECT_NAME)

@app.get("/health")
async def health_check():
    """
    Checks service status and hardware acceleration.
    """
    return {
        "status": "online",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }


@app.on_event("startup")
async def startup_event():
    print(f"🚀 {settings.PROJECT_NAME} starting on {settings.DEVICE.upper()}")

    # --- THIS IS THE MISSING PIECE ---
    # It creates the tables defined in models.py in the Postgres DB
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database schema initialized (Tables created).")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

    print(f"📍 Database: {settings.DATABASE_URL.split('@')[-1]}")
    print(f"📍 Vector Store: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
```

File: run_ingestion.py
```py
import asyncio
import os
import logging
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.vector import QdrantRepository
from app.db.graph import Neo4jRepository
from app.pipeline.ingestion import IngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 1. Directory in which the raw data sits
    raw_data_dir = settings.RAW_DATA_DIR

    # 2. Initialize the Qdrant Client (External Connection)
    logger.info("Connecting to Vector Database...")
    qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    qdrant_repo = QdrantRepository(client=qdrant_client)
    await qdrant_repo.initialize_collection()

    # 3. Initialize the Neo4j Client
    logger.info("Connecting to Graph Database...")
    neo4j_driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    graph_repo = Neo4jRepository(neo4j_driver)
    await graph_repo.setup_constraints()

    try:
        # 4. Initialize the Postgres Session (Unit of Work Pattern)
        async with AsyncSessionLocal.begin() as pg_session:

            # Inject ALL dependencies into the Facade
            service = IngestionService(
                pg_session=pg_session,
                qdrant_repo=qdrant_repo,
                graph_repo=graph_repo  # Added the missing injection here
            )

            # Execute the pipeline
            await service.process_directory(raw_data_dir)

    except Exception as e:
        logger.error(f"PIPELINE FAILED: Distributed transaction rolled back. Error: {e}")
    finally:
        # Clean up the network connection pools
        await qdrant_client.close()
        await neo4j_driver.close() # Added the missing cleanup here
        logger.info("Connections closed.")


if __name__ == "__main__":
    # Ensure Windows uses the correct asyncio event loop policy for certain networking tasks
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

File: test_enricher.py
```py
import asyncio
import json
from app.schemas.law_unit import LawUnitCreate
from app.core.custom_types import UnitType
from app.pipeline.ingestion.enricher import EnricherService


async def run_test():
    print("🚦 --- Starting GPU Enricher Test --- 🚦\n")

    # 1. Initialize the Service
    try:
        enricher = EnricherService()
        print("[OK] EnricherService initialized successfully.\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Enricher. Check config. Error: {e}")
        return

    # 2. Create Dummy Data (Simulating the output from your Parser)
    # We test a CHAPTER (should skip QuOTE) and an ARTICLE (should trigger QuOTE)
    chapter_unit = LawUnitCreate(
        id="cap_2",
        content="CAPITOLUL II: Vehiculele. Secțiunea 1: Condițiile privind circulația vehiculelor.",
        unit_type=UnitType.CHAPTER,
        metadata={"source": "test_script"}
    )

    article_unit = LawUnitCreate(
        id="art_14",
        content="Tramvaiele, tractoarele agricole sau forestiere, remorcile destinate a fi tractate de acestea, precum și troleibuzele se înregistrează la nivelul primăriilor comunelor, ale orașelor, ale municipiilor, ale sectoarelor municipiului București.",
        unit_type=UnitType.ARTICLE,
        metadata={"source": "test_script"}
    )

    units_to_test = [chapter_unit, article_unit]

    # 3. Run the Enricher
    print(f"🧠 Sending {len(units_to_test)} units to local Ollama models (Watch your GPU VRAM!)...\n")

    try:
        enriched_units = await enricher.enrich_batch(units_to_test)
    except Exception as e:
        print(f"[ERROR] Batch enrichment failed: {e}")
        return

    # 4. Display Results Visually
    print("\n✅ --- Enrichment Complete. Results: --- ✅\n")
    for eu in enriched_units:
        print(f"=== ID: {eu.id} | Type: {eu.unit_type.name} ===")
        print(f"Text: {eu.content}\n")

        # Print generated questions
        print("Generated Questions:")
        if eu.hypothetical_questions:
            print(json.dumps(eu.hypothetical_questions, indent=2, ensure_ascii=False))
        else:
            print("  [None - Skipped as expected for this unit type]")

        # Verify vector dimensions
        c_vec_len = len(eu.content_vector) if eu.content_vector else 0
        q_vec_count = len(eu.question_vectors) if eu.question_vectors else 0

        print(f"\nDiagnostics:")
        print(f" - Content Vector Dimension: {c_vec_len} (Should be 768 for nomic-embed-text)")
        print(f" - Number of Question Vectors: {q_vec_count} (Should match number of questions)")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    # Windows/WSL2 specific fix for asyncio loop policies if needed
    import sys

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_test())
```

File: tests\test_ingestion_pipeline.py
```py
import os
import sys
import pytest
import asyncio
from sqlalchemy import select, func

# 1. Setup the Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline.ingestion import run_ingestion_pipeline
from app.db.session import AsyncSessionLocal
from app.db.models import LawUnit
from app.core.custom_types import UnitType

RAW_DATA_DIR = "data/raw_text"


# This decorator tells pytest to spin up an event loop for this specific test
@pytest.mark.asyncio
async def test_ingestion_execution():
    """
    Integration Test:
    1. Checks if raw data exists.
    2. Runs the full ingestion pipeline (Parser -> DB).
    3. Connects to the DB to verify data was actually saved.
    """

    # A. Pre-check
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        with open(os.path.join(RAW_DATA_DIR, "dummy_test.txt"), "w", encoding="utf-8") as f:
            f.write("CAPITOLUL I: DISPOZITII GENERALE\nArt. 1.\n(1) Circulatia pe drumurile publice...")
        print(f"Created dummy test file in {RAW_DATA_DIR}")

    # B. Execution (Now awaited!)
    processed_units = await run_ingestion_pipeline(RAW_DATA_DIR)

    # C. Assertions (In-Memory)
    assert isinstance(processed_units, list)
    assert len(processed_units) > 0, "Pipeline returned empty list"

    print(f"\n[TEST] In-memory checks passed. Processed {len(processed_units)} units.")

    # D. Database Verification
    # Using the Context Manager pattern to safely open and close the async connection
    async with AsyncSessionLocal() as db:
        # 1. Check total count using SQLAlchemy 2.0 syntax
        stmt_count = select(func.count()).select_from(LawUnit)
        result_count = await db.execute(stmt_count)
        count = result_count.scalar_one()

        assert count > 0, "Database table 'law_units' is empty!"

        # 2. Check a specific unit type exists
        stmt_chapter = select(LawUnit).where(LawUnit.unit_type == UnitType.CHAPTER)
        result_chapter = await db.execute(stmt_chapter)
        chapter = result_chapter.scalar_one_or_none()

        assert chapter is not None, "No chapters found in DB"

        print(f"[TEST] Database verification passed. Found {count} rows in 'law_units'.")
        print(f"[TEST] Sample Chapter: {chapter.content}")


# The Entry Point Pattern for running the script directly (outside of pytest)
if __name__ == "__main__":
    asyncio.run(test_ingestion_execution())
```

