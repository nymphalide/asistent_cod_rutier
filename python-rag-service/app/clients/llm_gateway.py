import asyncio
from typing import List, Optional
from ollama import AsyncClient

from app.core.config import settings
from app.core.ai_registry import ModelRegistry


class LLMGateway:
    """
    Singleton Pattern: Ensures a single global instance.
    Gateway Pattern: Centralizes all external LLM API calls.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMGateway, cls).__new__(cls)
            cls._instance.client = AsyncClient(host=settings.OLLAMA_HOST)
            # ONE global bouncer to protect the GPU from OOM crashes
            cls._instance.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_ENRICHMENT_TASKS)
        return cls._instance

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