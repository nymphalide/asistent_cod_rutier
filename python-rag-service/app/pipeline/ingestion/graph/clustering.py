import logging
import asyncio
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import HDBSCAN
from collections import Counter
from pydantic import BaseModel, Field

from app.schemas.graph import (
    RawEntity, ConceptNode, CategoryNode, MentionsEdge, BelongsToEdge
)
from app.core.ai_registry import ModelRegistry
from app.clients.llm_gateway import LLMGateway

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
        self.gateway = LLMGateway()
        self.config = ModelRegistry.get_clustering_config()
        self.embedding_model = ModelRegistry.get_embedding_model()

    async def resolve_entities(self, raw_entities: List[RawEntity]) -> ClusteringResult:
        if not raw_entities:
            return ClusteringResult()

        # 1. Deduplicate
        unique_forms = list(set(e.surface_form for e in raw_entities))

        # 2. Embed via Gateway (Safely handles concurrent requests)
        tasks = [self.gateway.get_embedding(form, self.embedding_model) for form in unique_forms]
        embeddings = await asyncio.gather(*tasks)
        matrix = np.array(embeddings)

        # 3. Offload CPU-heavy synchronous math
        return await asyncio.to_thread(self._sync_resolve, raw_entities, unique_forms, matrix)

    def _sync_resolve(self, raw_entities: List[RawEntity], unique_forms: List[str],
                      matrix: np.ndarray) -> ClusteringResult:
        clusterer = HDBSCAN(
            min_cluster_size=self.config["min_cluster_size"],
            min_samples=self.config["min_samples"],
            metric=self.config["metric"]
        )
        labels = clusterer.fit_predict(matrix)

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

            result.concepts.append(ConceptNode(id=concept_id, name=canonical_name, surface_forms=list(set(forms))))
            category_set.add(category_name)
            result.belongs_to_edges.append(
                BelongsToEdge(source_concept_id=concept_id, target_category_name=category_name))

            for entity in raw_entities:
                if entity.surface_form in forms:
                    result.mentions_edges.append(
                        MentionsEdge(source_unit_id=entity.source_unit_id, target_concept_id=concept_id))

        result.categories = [CategoryNode(name=cat) for cat in category_set]
        return result

    def _get_dominant_category(self, canonical_name: str, raw_entities: List[RawEntity]) -> str:
        relevant = [e.category_label for e in raw_entities if e.surface_form == canonical_name]
        return Counter(relevant).most_common(1)[0][0] if relevant else "General"