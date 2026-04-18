import logging
import asyncio
import numpy as np
from typing import List, Dict
from sklearn.cluster import HDBSCAN
from collections import Counter
from pydantic import BaseModel, Field
from src.app.core.patterns import SingletonMeta

from src.app.schemas.graph import (
    RawEntity, ConceptNode, CategoryNode, MentionsEdge, BelongsToEdge
)
from src.app.core.ai_registry import ModelRegistry
from src.app.clients.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


class ClusteringResult(BaseModel):
    """DTO ensuring the Orchestrator receives the exact attributes it expects."""
    concepts: List[ConceptNode] = Field(default_factory=list)
    categories: List[CategoryNode] = Field(default_factory=list)
    mentions_edges: List[MentionsEdge] = Field(default_factory=list)
    belongs_to_edges: List[BelongsToEdge] = Field(default_factory=list)


class ClusteringEngine(metaclass=SingletonMeta):
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

        # We find the category for each word and prepend it to create a semantic anchor.
        # e.g., "Instituție: poliția rutieră" instead of just "poliția rutieră"
        prefixed_forms = [
            f"{self._get_dominant_category(form, raw_entities)}: {form}"
            for form in unique_forms
        ]


        # 2. Embed via Gateway (Safely handles concurrent requests)
        tasks = [self.gateway.get_embedding(prefixed_form, self.embedding_model) for prefixed_form in prefixed_forms]
        embeddings = await asyncio.gather(*tasks)
        matrix = np.array(embeddings)

        # 3. Offload CPU-heavy synchronous math
        # CRITICAL: We pass the clean `unique_forms` to `_sync_resolve`, not the `prefixed_forms`.
        # This ensures the AI uses the prefix for math, but Neo4j only saves the clean word.
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

            # FIX 1: ConceptNode does not use 'id' in your schema, only 'name' and 'surface_forms'
            result.concepts.append(ConceptNode(name=canonical_name, surface_forms=list(set(forms))))

            category_set.add(category_name)

            # FIX 2: BelongsToEdge requires 'source_concept_name', not an ID
            result.belongs_to_edges.append(
                BelongsToEdge(source_concept_name=canonical_name, target_category_name=category_name)
            )

            for entity in raw_entities:
                if entity.surface_form in forms:
                    # FIX 3: MentionsEdge requires 'target_concept_name' and 'extracted_text'
                    result.mentions_edges.append(
                        MentionsEdge(
                            source_unit_id=entity.source_unit_id,
                            target_concept_name=canonical_name,
                            extracted_text=entity.surface_form
                        )
                    )

        result.categories = [CategoryNode(name=cat) for cat in category_set]
        return result

    def _get_dominant_category(self, canonical_name: str, raw_entities: List[RawEntity]) -> str:
        relevant = [e.category_label for e in raw_entities if e.surface_form == canonical_name]
        return Counter(relevant).most_common(1)[0][0] if relevant else "General"