
# ============================================================
# FILE: app/__init__.py
# ============================================================



# ============================================================
# FILE: app/api/dependencies.py
# ============================================================

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that provides an async database session per request.
Implements the Unit of Work pattern: automatically commits on success,
and rolls back on any exception."""
    ...

# ============================================================
# FILE: app/clients/llm_gateway.py
# ============================================================

from typing import List, Optional


class LLMGateway:
    """
    Singleton Pattern: Ensures a single global instance.
    Gateway Pattern: Centralizes all external LLM API calls.
    """
    _instance = None

    def __new__(cls):
        ...

    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        ...

    async def generate_response(self, prompt: str, system_prompt: str=None) -> str:
        ...

# ============================================================
# FILE: app/clients/__init__.py
# ============================================================



# ============================================================
# FILE: app/core/ai_registry.py
# ============================================================

from typing import Dict, Any

class ModelRegistry:
    """
    Centralized registry for AI model hyperparameters.
    Separates behavioral AI logic from infrastructure environment variables.
    """

    @staticmethod
    def get_enricher_chat_config() -> Dict[str, Any]:
        """Configuration for the QuOTE strategy question generation."""
        ...

    @staticmethod
    def get_embedding_model() -> str:
        """The designated embedding model for dense vectors."""
        ...

    @staticmethod
    def get_embedding_dimensions() -> int:
        """Required by Qdrant to initialize the vector index."""
        ...

    @staticmethod
    def get_ner_config() -> Dict[str, Any]:
        """Configuration for the Zero-Shot NER extraction."""
        ...

    @staticmethod
    def get_clustering_config() -> Dict[str, Any]:
        """Hyperparameters for HDBSCAN and entity resolution."""
        ...

# ============================================================
# FILE: app/core/config.py
# ============================================================

from pydantic_settings import SettingsConfigDict
from pydantic import Field
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

class Settings(BaseSettings):
    """
    Configuration for the Python RAG microservice. [cite: 112]
    This class VALIDATES the environment; it does not set the truth.
    """
    PROJECT_NAME: str = 'Optimised Romanian Traffic Code RAG'
    RAW_DATA_DIR: str = str(PROJECT_ROOT / 'data' / 'raw_text')
    DATABASE_URL: str = Field(validation_alias='PYTHON_DATABASE_URL')
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_COLLECTION: str = 'traffic_code_vectors'
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    OLLAMA_HOST: str = 'http://localhost:11434'
    MAX_CONCURRENT_ENRICHMENT_TASKS: int = 5
    USE_CUDA: bool = True

    @property
    def DEVICE(self) -> str:
        """Dynamically selects the GPU (RTX 4060) if available."""
        ...
    model_config = SettingsConfigDict(case_sensitive=True, extra='ignore')
settings = Settings()

# ============================================================
# FILE: app/core/custom_types.py
# ============================================================

from enum import Enum

class UnitType(str, Enum):
    CHAPTER = 'chapter'
    SECTION = 'section'
    ARTICLE = 'article'
    PROLOGUE = 'prologue'
    PARAGRAPH = 'paragraph'
    LETTER_ITEM = 'letter_item'
    NUMBERED_ITEM = 'num_item'

# ============================================================
# FILE: app/core/worker_app.py
# ============================================================

import procrastinate
from src.app import settings
dsn = settings.DATABASE_URL.replace('+asyncpg', '')
task_app = procrastinate.App(connector=procrastinate.PsycopgConnector(conninfo=dsn), import_paths=['app.pipeline.ingestion.tasks'])

# ============================================================
# FILE: app/db/graph.py
# ============================================================

import logging
from typing import List, Dict, Any
from neo4j import AsyncDriver, AsyncTransaction
from src.app import GraphPayload
logger = logging.getLogger(__name__)

class Neo4jRepository:
    """
    Repository Pattern: Translates Pydantic domain models into Neo4j graph structures.
    Uses strict Dependency Injection for the driver to remain infrastructure-agnostic.
    """

    def __init__(self, driver: AsyncDriver):
        ...

    async def setup_constraints(self) -> None:
        """Ensures unique constraints exist for lightning-fast MERGE operations.
Must be called once during application startup."""
        ...

    async def upsert_payload(self, payload: GraphPayload) -> None:
        """Unit of Work Pattern: Executes the entire GraphPayload within a single transaction.
If any part fails, the entire batch rolls back to prevent orphaned nodes."""
        ...

    async def _execute_batch_upsert(self, tx: AsyncTransaction, payload: GraphPayload) -> None:
        """Executes the optimized UNWIND queries sequentially.
Nodes MUST be created before Edges."""
        ...

    async def _upsert_law_units(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_concepts(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_categories(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_external_laws(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_reference_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_part_of_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_mentions_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_belongs_to_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

    async def _upsert_refers_to_external_edges(self, tx: AsyncTransaction, data: List[Dict[str, Any]]) -> None:
        ...

# ============================================================
# FILE: app/db/models.py
# ============================================================

from typing import Optional, List, Any, Dict
from sqlalchemy import String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, ENUM
from src.app import Base
from src.app import UnitType


class LawUnit(Base):
    """
    SQLAlchemy model representing the 'law_units' table.
    Stores the graph nodes (Articles, Paragraphs) and their vector embeddings.
    """
    __tablename__ = 'law_units'
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    parent_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey('law_units.id', ondelete='CASCADE'), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    unit_type: Mapped[UnitType] = mapped_column(ENUM(UnitType, name='unit_type_enum', create_type=True, metadata=Base.metadata), nullable=False)
    meta_info: Mapped[Dict[str, Any]] = mapped_column('meta_info', JSONB, server_default='{}')
    children: Mapped[List['LawUnit']] = relationship('LawUnit', back_populates='parent', foreign_keys=[parent_id], cascade='all, delete-orphan')
    parent: Mapped[Optional['LawUnit']] = relationship('LawUnit', remote_side=[id], foreign_keys=[parent_id], back_populates='children')

    def __repr__(self):
        ...

# ============================================================
# FILE: app/db/repository.py
# ============================================================

from typing import List, Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from src.app import LawUnit
from src.app import LawUnitCreate

class LawUnitRepository:
    """
    Repository for handling LawUnit (Traffic Code) database operations.
    Acts as the 'Librarian', abstracting SQL away from the business logic.
    """

    def __init__(self, db: AsyncSession):
        ...

    async def create(self, unit_in: LawUnitCreate) -> LawUnit:
        """Creates a single LawUnit record by converting the Pydantic
schema into a SQLAlchemy model instance."""
        ...

    async def get(self, unit_id: str) -> Optional[LawUnit]:
        """Retrieves a LawUnit by its ID using async execute."""
        ...

    async def get_children(self, parent_id: str) -> Sequence[LawUnit]:
        """Retrieves all direct children of a specific unit using async execute."""
        ...

    async def bulk_upsert(self, units_in: List[LawUnitCreate]) -> int:
        ...

# ============================================================
# FILE: app/db/session.py
# ============================================================

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from src.app import settings
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
Base = declarative_base()

# ============================================================
# FILE: app/db/vector.py
# ============================================================

import uuid
import logging
from typing import List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.app import LawUnitEnriched

logger = logging.getLogger(__name__)
NAMESPACE_RAG = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

class QdrantRepository:
    """
    Repository layer for interacting with Qdrant.
    Uses Dependency Injection for connection pooling and implements
    strict error handling for distributed transactions.
    """

    def __init__(self, client: AsyncQdrantClient):
        ...

    async def initialize_collection(self) -> None:
        ...

    def _map_to_points(self, unit: LawUnitEnriched) -> List[models.PointStruct]:
        ...

    async def bulk_upsert(self, units: List[LawUnitEnriched]) -> None:
        ...

# ============================================================
# FILE: app/db/__init__.py
# ============================================================

from .session import Base, AsyncSessionLocal, engine
from .models import LawUnit
from .repository import LawUnitRepository

# ============================================================
# FILE: app/pipeline/__init__.py
# ============================================================



# ============================================================
# FILE: app/pipeline/ingestion/enricher.py
# ============================================================

import logging
from typing import List
from src.app import LawUnitCreate, LawUnitEnriched

logger = logging.getLogger(__name__)

class EnricherService:
    """
    Service Layer responsible for AI enrichment (Embeddings & QuOTE Strategy).
    Delegates infrastructure calls to the LLMGateway.
    """

    def __init__(self):
        ...

    async def _generate_questions(self, content: str) -> List[str]:
        ...

    async def enrich_unit(self, unit: LawUnitCreate) -> LawUnitEnriched:
        ...

    async def enrich_batch(self, units: List[LawUnitCreate]) -> List[LawUnitEnriched]:
        ...

# ============================================================
# FILE: app/pipeline/ingestion/parser.py
# ============================================================

import re
from typing import List, Any, Tuple
from src.app import LawUnitCreate


class TrafficCodeParser:
    REGEX_CHAPTER = re.compile('^(CAPITOLUL|TITLUL)\\s+([IVXLCDM]+)(?::\\s*(.*))?', re.IGNORECASE)
    REGEX_SECTION = re.compile('^SEC[TȚ]IUNEA\\s+(?:nr\\.\\s*)?(\\d+|a\\s+[a-z0-9-]+-a)(?:\\s*(.*))?', re.IGNORECASE)
    REGEX_ART = re.compile('^Art\\.\\s*(\\d+(?:\\.\\d+)?)\\.?\\s*(.*)', re.IGNORECASE)
    REGEX_PARA = re.compile('^\\((\\d+(?:\\.\\d+)?)\\)\\s*(.*)')
    REGEX_LETTER = re.compile('^([a-z])\\)\\s*(.*)')
    REGEX_NUM_ITEM = re.compile('^(\\d+(?:\\.\\d+)?)\\.\\s*(.*?)(?:\\s*-\\s*(.*))?$')

    def __init__(self, file_path: str):
        ...

    def _read_file_safely(self, file_path: str) -> List[str]:
        ...

    def _flush_buffer(self):
        ...

    def _tokenize(self, line: str) -> Tuple[str, Any]:
        """The Lexer: Returns a structural tuple (Token_Type, Match_Data)"""
        ...

    def parse(self) -> List[LawUnitCreate]:
        ...

# ============================================================
# FILE: app/pipeline/ingestion/tasks.py
# ============================================================

import logging
import signal
from typing import List
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
from src.app import task_app
from src.app import settings
from src.app.pipeline.ingestion.graph import DeterministicExtractor
from src.app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from src.app.pipeline.ingestion.graph import ClusteringEngine

logger = logging.getLogger(__name__)
GLOBAL_DETERMINISTIC_EXTRACTOR = DeterministicExtractor()
GLOBAL_SEMANTIC_EXTRACTOR = SemanticExtractor()
GLOBAL_CLUSTERING_ENGINE = ClusteringEngine()
GLOBAL_QDRANT_CLIENT = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
GLOBAL_NEO4J_DRIVER = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

@task_app.task(name='ingest_vectors_batch')
async def ingest_vectors_batch_task(unit_ids: List[str]):
    """TASK A: Handles Ollama embeddings and Qdrant (The Heavy Lifter)"""
    ...

@task_app.task(name='ingest_graph_batch')
async def ingest_graph_batch_task(unit_ids: List[str]):
    """TASK B: Handles GLiNER, HDBSCAN, and Neo4j (The Fast/Volatile Lifter)"""
    ...

async def shutdown_connections():
    """Gracefully closes global connection pools."""
    ...

def handle_sigterm(signum, frame):
    """Catches termination signals (Ctrl+C) cross-platform."""
    ...
signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

# ============================================================
# FILE: app/pipeline/ingestion/__init__.py
# ============================================================

import logging
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Facade Pattern for orchestrating the RAG ETL process.
    Implements the Transactional Outbox Pattern to decouple databases.
    """

    def __init__(self, pg_session: AsyncSession):
        ...

    async def process_directory(self, raw_data_dir: str, trigger_tasks: bool=True) -> None:
        ...

# ============================================================
# FILE: app/pipeline/ingestion/graph/clustering.py
# ============================================================

import logging
import numpy as np
from typing import List
from pydantic import BaseModel, Field
from src.app import RawEntity, ConceptNode, CategoryNode, MentionsEdge, BelongsToEdge

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
    Relies on the Singleton LLMGateway for safe embedding generation.
    """

    def __init__(self):
        ...

    async def resolve_entities(self, raw_entities: List[RawEntity]) -> ClusteringResult:
        ...

    def _sync_resolve(self, raw_entities: List[RawEntity], unique_forms: List[str], matrix: np.ndarray) -> ClusteringResult:
        ...

    def _get_dominant_category(self, canonical_name: str, raw_entities: List[RawEntity]) -> str:
        ...

# ============================================================
# FILE: app/pipeline/ingestion/graph/orchestrator.py
# ============================================================

import logging
from typing import List
from src.app.db.graph import Neo4jRepository
from src.app.pipeline.ingestion.graph import DeterministicExtractor
from src.app.pipeline.ingestion.graph.extractors.semantic import SemanticExtractor
from src.app.pipeline.ingestion.graph import ClusteringEngine
from src.app import LawUnit
logger = logging.getLogger(__name__)

class GraphOrchestrator:
    """
    Facade Pattern: Manages the Knowledge Graph ETL workflow.
    Now strictly relies on in-memory domain models, removing circular DB dependencies.
    """

    def __init__(self, graph_repo: Neo4jRepository, deterministic_extractor: DeterministicExtractor, semantic_extractor: SemanticExtractor, clustering_engine: ClusteringEngine):
        ...

    async def process_batch(self, units: List[LawUnit]) -> None:
        """Executes the full graph formulation pipeline for a batch of pre-loaded LawUnits."""
        ...

# ============================================================
# FILE: app/pipeline/ingestion/graph/__init__.py
# ============================================================



# ============================================================
# FILE: app/pipeline/ingestion/graph/extractors/deterministic.py
# ============================================================

import re
import logging
from typing import Optional
from pydantic import BaseModel, Field
from src.app import ReferenceEdge, ExternalLawNode, RefersToExternalEdge
logger = logging.getLogger(__name__)

class DeterministicResult(BaseModel):
    """
    Packages the Layer 1 graph boundaries for the Orchestrator.
    Prevents the Orchestrator from needing to know how the regex works.
    """
    internal_edges: list[ReferenceEdge] = Field(default_factory=list)
    external_nodes: list[ExternalLawNode] = Field(default_factory=list)
    external_edges: list[RefersToExternalEdge] = Field(default_factory=list)

class DeterministicExtractor:
    """
    Parses legal text to deterministically build the Layer 1
    structural Knowledge Graph skeleton.
    """
    REGEX_EXTERNAL = re.compile('(?P<type>Legea|Ordonan[tț][aă](?:\\s+Guvernului|\\s+de\\s+urgen[tț][aă])?|Regulamentul\\s*\\(UE\\))\\s+nr\\.\\s*(?P<num>\\d+/\\d{4})', re.IGNORECASE)
    REGEX_INTERNAL = re.compile('(?P<abs_art_range>art\\.\\s*(?P<r_start>\\d+(?:\\.\\d+)?)\\s*-\\s*(?P<r_end>\\d+(?:\\.\\d+)?))|(?P<abs_art_alin>art\\.\\s*(?P<aa_art>\\d+(?:\\.\\d+)?)\\s*alin\\.\\s*\\((?P<aa_alin>[a-z0-9.]+)\\))|(?P<abs_art>art\\.\\s*(?P<a_art>\\d+(?:\\.\\d+)?))|(?P<rel_alin>alin\\.\\s*\\((?P<r_alin>[a-z0-9.]+)\\))', re.IGNORECASE)

    def extract_references(self, source_id: str, content: str) -> DeterministicResult:
        """Main execution method for deterministic parsing."""
        ...

    def _parse_external(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds boundaries triggering external routing strategies."""
        ...

    def _parse_internal(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds internal nodes to build standard traversal paths."""
        ...

    def _get_base_article(self, source_id: str) -> Optional[str]:
        """Extracts the parent article ID to resolve relative references.
Example: 'art_102_alin_3_lit_a' -> 'art_102'"""
        ...

# ============================================================
# FILE: app/pipeline/ingestion/graph/extractors/semantic.py
# ============================================================

import logging
from typing import List
from gliner import GLiNER
from src.app import RawEntity
from src.app import LawUnit

logger = logging.getLogger(__name__)

class SemanticExtractor:

    def __init__(self, ontology_path: str='app/pipeline/ingestion/graph/ontology.json'):
        ...

    def _load_ontology(self) -> List[str]:
        ...

    def _get_model(self) -> GLiNER:
        """Lazy loader utilizing the Double-Checked Locking Pattern.
Ensures thread-safe Singleton initialization for the GLiNER model."""
        ...

    def _sync_predict_batch(self, units: List[LawUnit]) -> List[RawEntity]:
        ...

    async def extract_batch(self, units: List[LawUnit]) -> List[RawEntity]:
        ...

# ============================================================
# FILE: app/pipeline/ingestion/graph/extractors/__init__.py
# ============================================================



# ============================================================
# FILE: app/schemas/graph.py
# ============================================================

from typing import List, Set
from pydantic import BaseModel, Field
from src.app import UnitType

class RawEntity(BaseModel):
    """
    Output from the GLiNER Semantic Extractor before clustering.
    Captures the exact string and its predicted category.
    """
    surface_form: str = Field(..., description="Exact string found in text (e.g., 'amenzii')")
    category_label: str = Field(..., description="Ontology label assigned by GLiNER (e.g., 'Sancțiuni')")
    source_unit_id: str = Field(..., description='The LawUnit ID where this was found')

class LawUnitNode(BaseModel):
    """
    Layer 1: The Skeleton.
    A lightweight routing node mapped 1:1 with Postgres.
    Excludes the heavy 'content' field to optimize Graph RAM.
    """
    id: str = Field(..., description="Canonical ID from Postgres (e.g., 'art_5')")
    unit_type: UnitType = Field(..., description='Enum defining the hierarchy level')

class ConceptNode(BaseModel):
    """
    Layer 2: The Canonical Concept post-HDBSCAN clustering.
    """
    name: str = Field(..., description='The mathematical centroid name')
    surface_forms: Set[str] = Field(default_factory=set, description='Deduplicated array of all exact string variations')

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

class ReferenceEdge(BaseModel):
    """Layer 1: Deterministic cross-reference ([:REFERENCES])."""
    source_id: str = Field(..., description='ID of the LawUnit making the citation')
    target_id: str = Field(..., description='ID of the referenced LawUnit')

class PartOfEdge(BaseModel):
    """Layer 1 Hierarchy: Structural parent-child relationship ([:PART_OF])."""
    child_id: str = Field(..., description='ID of the subordinate LawUnit (e.g., Paragraph)')
    parent_id: str = Field(..., description='ID of the parent LawUnit (e.g., Article)')

class MentionsEdge(BaseModel):
    """Layer 2: Semantic bridge from text to concept ([:MENTIONS])."""
    source_unit_id: str = Field(..., description='ID of the LawUnit')
    target_concept_name: str = Field(..., description='Name of the Canonical Concept')
    extracted_text: str = Field(..., description='The specific surface form that triggered this connection')

class BelongsToEdge(BaseModel):
    """Layer 3: Concept classification ([:BELONGS_TO])."""
    source_concept_name: str = Field(..., description='Name of the Canonical Concept')
    target_category_name: str = Field(..., description='Name of the Category')

class RefersToExternalEdge(BaseModel):
    """
    Layer 1 Boundary: LawUnit -> ExternalLawNode connection ([:REFERS_TO_EXTERNAL]).
    """
    source_unit_id: str = Field(..., description='ID of the internal LawUnit making the citation')
    target_external_id: str = Field(..., description='Normalized ID of the ExternalLawNode')

class GraphPayload(BaseModel):
    """
    The final validated package the Orchestrator hands to the Neo4j Adapter.
    Enforces a strict contract between the ML pipeline and the database layer.
    """
    law_units: List[LawUnitNode] = Field(default_factory=list)
    concepts: List[ConceptNode] = Field(default_factory=list)
    categories: List[CategoryNode] = Field(default_factory=list)
    external_laws: List[ExternalLawNode] = Field(default_factory=list)
    reference_edges: List[ReferenceEdge] = Field(default_factory=list)
    part_of_edges: List[PartOfEdge] = Field(default_factory=list)
    mentions_edges: List[MentionsEdge] = Field(default_factory=list)
    belongs_to_edges: List[BelongsToEdge] = Field(default_factory=list)
    refers_to_external_edges: List[RefersToExternalEdge] = Field(default_factory=list)

# ============================================================
# FILE: app/schemas/law_unit.py
# ============================================================

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field
from src.app import UnitType

class LawUnitBase(BaseModel):
    id: str = Field(..., description="Canonical ID (e.g., 'art_102_alin_2')")
    content: str = Field(..., description='The actual legal text')
    unit_type: UnitType
    meta_info: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(from_attributes=True)

class LawUnitCreate(LawUnitBase):
    parent_id: Optional[str] = None

class LawUnitEnriched(LawUnitCreate):
    """
    DTO carrying the AI-enriched data destined for Qdrant.
    """
    hypothetical_questions: List[str] = Field(default_factory=list, description='Distinct questions generated by the Ollama LLM')
    content_vector: Optional[List[float]] = Field(default=None, description='Dense embedding of the raw legal content')
    question_vectors: Optional[List[List[float]]] = Field(default=None, description='List of dense embeddings, mapping 1:1 to hypothetical_questions')

class LawUnitResponse(LawUnitBase):
    parent_id: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)
