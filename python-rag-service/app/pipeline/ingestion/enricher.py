import json
import logging
import asyncio
from typing import List, Optional

from app.schemas.law_unit import LawUnitCreate, LawUnitEnriched
from app.core.custom_types import UnitType
from app.core.ai_registry import ModelRegistry
from app.clients.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

class EnricherService:
    """
    Service Layer responsible for AI enrichment (Embeddings & QuOTE Strategy).
    Delegates infrastructure calls to the LLMGateway.
    """
    def __init__(self):
        self.gateway = LLMGateway()
        self.embedding_model = ModelRegistry.get_embedding_model()

    async def _generate_questions(self, content: str) -> List[str]:
        prompt = (
            "You are a Romanian legal expert. Read the following traffic code text "
            "and generate exactly 3 hypothetical questions that a driver might ask, "
            "which are directly answered by this text. "
            "Respond ONLY in valid JSON format using the following schema: "
            '{"questions": ["question 1", "question 2", "question 3"]}\n\n'
            f"Text: {content}"
        )

        try:
            response_text = await self.gateway.generate_response(prompt=prompt)
            result = json.loads(response_text)
            return result.get("questions", [])[:3]
        except Exception as e:
            logger.error(f"LLM Question Generation failed: {e}")
            return []

    async def enrich_unit(self, unit: LawUnitCreate) -> LawUnitEnriched:
        # 1. Skip QuOTE for Structural Headers
        if unit.unit_type in [UnitType.CHAPTER, UnitType.SECTION]:
            content_vector = await self.gateway.get_embedding(unit.content, self.embedding_model)
            return LawUnitEnriched(
                **unit.model_dump(by_alias=True),
                hypothetical_questions=[],
                content_vector=content_vector,
                question_vectors=[]
            )

        # 2. Generate QuOTE Questions
        questions = await self._generate_questions(unit.content)

        # 3. Parallel Vector Generation (Content + N Questions)
        # The LLMGateway inherently throttles these concurrent tasks safely
        tasks = [self.gateway.get_embedding(unit.content, self.embedding_model)]
        for q in questions:
            tasks.append(self.gateway.get_embedding(q, self.embedding_model))

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