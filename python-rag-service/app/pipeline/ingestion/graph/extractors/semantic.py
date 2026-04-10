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

        # +++ THE FIX: Lazy Loading +++
        # We start with None. The model will NOT download when this class is instantiated.
        self._model = None

    def _load_ontology(self) -> List[str]:
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f).get("labels", [])

    def _get_model(self) -> GLiNER:
        """Lazy loader: only boots GLiNER when actual extraction is requested."""
        if self._model is None:
            logger.info(f"Initializing {self.model_name} on {self.device}...")
            try:
                self._model = GLiNER.from_pretrained(self.model_name).to(self.device)
            except Exception as e:
                logger.error(f"Model initialization failed: {e}")
                raise
        return self._model

    def _sync_predict_batch(self, units: List[LawUnit]) -> List[RawEntity]:
        texts = [unit.content for unit in units]
        raw_entities: List[RawEntity] = []

        # Grab the model. This triggers the download/GPU load ONLY the very first time.
        model = self._get_model()

        batch_results = model.batch_predict_entities(
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