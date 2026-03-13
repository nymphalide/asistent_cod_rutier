from typing import List, Set, Optional
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


# ==========================================
# MASTER FACADE PAYLOAD
# ==========================================

class GraphPayload(BaseModel):
    """
    The final validated package the Orchestrator hands to the Neo4j Adapter.
    Enforces a strict contract between the ML pipeline and the database layer.
    """
    law_units: List[LawUnitNode] = Field(default_factory=list)
    concepts: List[ConceptNode] = Field(default_factory=list)
    categories: List[CategoryNode] = Field(default_factory=list)

    reference_edges: List[ReferenceEdge] = Field(default_factory=list)
    part_of_edges: List[PartOfEdge] = Field(default_factory=list)
    mentions_edges: List[MentionsEdge] = Field(default_factory=list)
    belongs_to_edges: List[BelongsToEdge] = Field(default_factory=list)