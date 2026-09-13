from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

import httpx
from openai import AsyncOpenAI, OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Explicit HTTP timeouts to prevent hanging on slow upstream responses
DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=12.0,   # Total request timeout in seconds
    connect=4.0,   # Max time to establish TCP/TLS connection
    read=8.0,      # Max time between streamed chunk reads
    write=4.0,     # Max time to write request payload
)

# Initialize asynchronous OpenAI client with Google AI Studio's base URL and explicit timeouts
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=DEFAULT_TIMEOUT,
    max_retries=1,
)

# Synchronous client instance (with identical timeout) for fallback/sync consumers
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=DEFAULT_TIMEOUT,
    max_retries=1,
)


def generate_mock_embedding(text: str, dim: int = 3072) -> list[float]:
    """
    Generates a deterministic unit-normalized mock embedding vector for local testing.
    Identical queries produce identical vectors (allowing semantic cache matching)
    without invoking external API quota or incurring network latency.
    """
    import hashlib
    import math
    import random

    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    raw_vector = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw_vector)) or 1.0
    return [round(x / norm, 6) for x in raw_vector]


class LLMService:
    """
    Service for interacting with OpenAI-compatible APIs (Google AI Studio) asynchronously,
    with strict timeouts, non-blocking streaming, and optional mock embedding mode.
    """

    def __init__(self, custom_client: Optional[AsyncOpenAI] = None) -> None:
        """
        Initialize the LLMService.

        Args:
            custom_client: Optional AsyncOpenAI client instance.
        """
        self.client = custom_client or async_client

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for the input text.
        If USE_MOCK_EMBEDDINGS is enabled, returns a deterministic mock vector
        immediately without making external network calls.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats representing the vector embedding.
        """
        if settings.USE_MOCK_EMBEDDINGS:
            logger.info("USE_MOCK_EMBEDDINGS is active; using mock embedding for text.")
            return generate_mock_embedding(text, dim=settings.EMBEDDING_DIMENSIONS)

        response = await self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def create_chat_stream(self, query: str):
        """
        Initiates the chat completion stream from the upstream LLM.
        Fails fast if connection cannot be established within timeout.
        """
        return await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and concise AI assistant.",
                },
                {"role": "user", "content": query},
            ],
            stream=True,
        )

    async def stream_chat_completion(self, query: str) -> AsyncGenerator[str, None]:
        """
        Streams chat completions from the configured model as an async generator,
        yielding response text chunks as they arrive.
        """
        response_stream = await self.create_chat_stream(query)

        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Singleton instance using the non-blocking AsyncOpenAI client
llm_service = LLMService(async_client)


# Convenience module-level functions
async def get_embedding(text: str) -> list[float]:
    return await llm_service.get_embedding(text)


async def stream_chat_completion(query: str) -> AsyncGenerator[str, None]:
    async for chunk in llm_service.stream_chat_completion(query):
        yield chunk
