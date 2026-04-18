import asyncio
from typing import List, Optional
from ollama import AsyncClient

from src.app.core.config import settings
from src.app.core.ai_registry import ModelRegistry
from src.app.core.patterns import SingletonMeta


class LLMGateway(metaclass=SingletonMeta):
    """
    Gateway Pattern: Centralizes all external LLM API calls.
    SingletonMeta ensures only one connection pool exists.
    """

    def __init__(self):
        # Because of the metaclass, this is guaranteed to run exactly once.
        self.client = AsyncClient(host=settings.OLLAMA_HOST)
        self._gen_semaphore = None
        self._embed_semaphore = None

    @property
    def gen_semaphore(self):
        # Lazy-load inside the active event loop to prevent RuntimeError
        if self._gen_semaphore is None:
            self._gen_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GENERATION_TASKS)
        return self._gen_semaphore

    @property
    def embed_semaphore(self):
        if self._embed_semaphore is None:
            self._embed_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_EMBEDDING_TASKS)
        return self._embed_semaphore

    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        if not text.strip():
            return None

        # Route through the fast embedding line
        async with self.embed_semaphore:
            response = await self.client.embeddings(model=model, prompt=text)
            return response['embedding']

    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        config = ModelRegistry.get_enricher_chat_config()
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # Route through the heavy generation line
        async with self.gen_semaphore:
            response = await self.client.chat(messages=messages, **config)
            return response['message']['content']