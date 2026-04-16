import asyncio
from typing import List, Optional
from ollama import AsyncClient

from app.core.config import settings
from app.core.ai_registry import ModelRegistry
from app.core.patterns import SingletonMeta


class LLMGateway(metaclass=SingletonMeta):
    """
    Gateway Pattern: Centralizes all external LLM API calls.
    SingletonMeta ensures only one connection pool exists.
    """

    def __init__(self):
        # Because of the metaclass, this is guaranteed to run exactly once.
        self.client = AsyncClient(host=settings.OLLAMA_HOST)
        self._semaphore = None

    @property
    def semaphore(self):
        # Lazy-load inside the active event loop to prevent RuntimeError
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_ENRICHMENT_TASKS)
        return self._semaphore

    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        if not text.strip():
            return None

        async with self.semaphore:
            response = await self.client.embeddings(model=model, prompt=text)
            return response['embedding']

    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        config = ModelRegistry.get_enricher_chat_config()
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        async with self.semaphore:
            response = await self.client.chat(messages=messages, **config)
            return response['message']['content']