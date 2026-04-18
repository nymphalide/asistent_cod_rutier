from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """
    The standardized request payload coming from the Orchestrator/Router.
    Now supports both Deterministic and Semantic graph traversals.
    """
    # --- 1. Vector Search Inputs ---
    query_text: Optional[str] = Field(None, description="The natural language query (for Qdrant/BM25)")
    metadata_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-filters (e.g., {'category': 'Sancțiuni'})"
    )

    # --- 2. Deterministic Graph Inputs (Your Original Design) ---
    target_unit_id: Optional[str] = Field(None, description="A specific node ID to start traversal from (e.g., 'art_102')")
    max_hops: int = Field(default=1, description="Depth of graph traversal. 1=Parent/Child, 2=References, 3=Semantic")
    allowed_edge_types: List[str] = Field(
        default_factory=lambda: ["PART_OF"],
        description="Specific Neo4j edges allowed for this hop"
    )

    # --- 3. Semantic Graph Inputs (The New SOTA Addition) ---
    extracted_concepts: List[str] = Field(
        default_factory=list,
        description="Exact Neo4j ConceptNode names extracted via GLiNER (e.g., ['amenda', 'viteza'])"
    )

    # --- 4. Execution Constraints ---
    top_k: int = Field(default=5, description="Number of results to return per strategy before fusion")


class RetrievedChunk(BaseModel):
    """
    The uniform output from ANY search strategy, heavily modified
    to support Reciprocal Rank Fusion and Cross-Encoding.
    """
    unit_id: str = Field(..., description="Canonical ID from Postgres (e.g., 'art_102')")
    content: str = Field(..., description="The text chunk")
    parent_id: Optional[str] = Field(None, description="Used for stitching paragraphs back into Articles")
    source_strategy: str = Field(..., description="E.g., 'qdrant_hybrid', 'neo4j_semantic_walk'")

    # --- The Scoring Pipeline ---
    raw_score: float = Field(
        default=0.0,
        description="The native DB score (Cosine similarity, BM25, or Graph proximity)"
    )
    fusion_score: float = Field(
        default=0.0,
        description="The calculated RRF score after merging streams"
    )
    cross_encoder_score: float = Field(
        default=0.0,
        description="The final GPU-calculated relevance logit"
    )