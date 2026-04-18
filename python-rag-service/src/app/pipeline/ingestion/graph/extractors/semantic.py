import json
import logging
import asyncio
from typing import List
from gliner import GLiNER # type: ignore
import threading
from pathlib import Path

from src.app.schemas.graph import RawEntity
from src.app.schemas.law_unit import LawUnitCreate
from src.app.core.config import settings
from src.app.core.ai_registry import ModelRegistry
from src.app.core.patterns import SingletonMeta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SemanticExtractor(metaclass=SingletonMeta):
    def __init__(self, ontology_path: str = None):
        self.ontology_path = ontology_path or settings.ONTOLOGY_PATH
        self.labels = self._load_ontology()

        # Pulling behavioral config from Registry, infrastructure from Settings
        config = ModelRegistry.get_ner_config()
        self.model_name = config["model"]
        self.batch_size = config["batch_size"]
        self.device = settings.DEVICE

        # Lazy Loading
        # We start with None. The model will NOT download when this class is instantiated.
        self._model = None

        # Initialize a threading lock to prevent simultaneous VRAM allocations
        self._lock = threading.Lock()

        self._executor = ThreadPoolExecutor(
            max_workers=config.get("max_concurrent_threads", 1),
            thread_name_prefix="gliner_inference"
        )

    def _load_ontology(self) -> List[str]:
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f).get("labels", [])

    def _get_model(self) -> GLiNER:
        """
        Lazy loader utilizing the Double-Checked Locking Pattern.
        Ensures thread-safe Singleton initialization for the GLiNER model.
        """
        # 1st Check: Fast path. If the model is already loaded, skip lock overhead.
        if self._model is None:
            # Acquire lock: Only one thread can enter this block at a time.
            with self._lock:
                # 2nd Check: Ensures waiting threads don't reload the model
                # after the first thread finishes and releases the lock.
                if self._model is None:
                    logger.info(f"Initializing {self.model_name} on {self.device}...")
                    try:
                        self._model = GLiNER.from_pretrained(self.model_name).to(self.device)
                    except Exception as e:
                        logger.error(f"Model initialization failed: {e}")
                        raise
        return self._model

    def _sync_predict_batch(self, units: List[LawUnitCreate]) -> List[RawEntity]:
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

    async def extract_batch(self, units: List[LawUnitCreate]) -> List[RawEntity]:
        loop = asyncio.get_running_loop()
        # Route the heavy ML task explicitly to our constrained, dedicated pool
        return await loop.run_in_executor(self._executor, self._sync_predict_batch, units)