from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """
    The standardized request payload coming from the Orchestrator/Router.
    """
    query_text: Optional[str] = Field(None, description="The natural language query (for Qdrant/BM25)")
    target_unit_id: Optional[str] = Field(None, description="A specific node ID to start traversal from (for Neo4j)")
    top_k: int = Field(default=5, description="Number of results to return per strategy")
    metadata_filters: Dict[str, Any] = Field(default_factory=dict,
                                             description="Pre-filters (e.g., {'category': 'Sancțiuni'})")

    # Traversal Budget
    max_hops: int = Field(default=1, description="Depth of graph traversal. 1=Parent/Child, 2=References, 3=Semantic")
    allowed_edge_types: List[str] = Field(
        default_factory=lambda: ["PART_OF"],
        description="Specific Neo4j edges allowed for this hop"
    )


class RetrievedChunk(BaseModel):
    """
    The uniform output from ANY search strategy.
    """
    unit_id: str = Field(..., description="Canonical ID from Postgres (e.g., 'art_102')")
    content: str = Field(..., description="The text chunk to be fed to the LLM")
    score: float = Field(default=0.0, description="Raw confidence score from the underlying DB")
    source_strategy: str = Field(..., description="Which adapter found this (e.g., 'qdrant_dense', 'neo4j_graph')")
    parent_id: Optional[str] = Field(None, description="Used to stitch paragraphs back into full Articles")