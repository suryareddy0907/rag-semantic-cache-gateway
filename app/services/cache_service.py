from __future__ import annotations

import json
import logging
from typing import Optional

from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.vector_cache import SemanticCache

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    Service responsible for managing semantic vector caching in PostgreSQL (pgvector)
    and low-latency caching in Redis.
    """

    def __init__(self, db: Session, redis_client: Optional[Redis] = None) -> None:
        """
        Initialize the SemanticCacheService.

        Args:
            db: Active SQLAlchemy database session.
            redis_client: Optional Redis client instance for fast caching.
        """
        self.db = db
        self.redis = redis_client

    def get_similar_cached_response(
        self, query_embedding: list[float], threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> str | None:
        """
        Queries PostgreSQL using pgvector cosine distance to find a cached query
        matching the similarity threshold.

        In pgvector, cosine_distance = 1.0 - cosine_similarity.
        Therefore, similarity >= threshold corresponds to:
            cosine_distance <= (1.0 - threshold)

        Args:
            query_embedding: The vector embedding list of the input query.
            threshold: Cosine similarity threshold (e.g. 0.85).

        Returns:
            The cached response_text if a similar query is found within threshold,
            otherwise None.
        """
        max_distance = 1.0 - threshold
        cosine_distance_expr = SemanticCache.embedding.cosine_distance(query_embedding)

        match = (
            self.db.query(SemanticCache)
            .filter(cosine_distance_expr <= max_distance)
            .order_by(cosine_distance_expr.asc())
            .first()
        )

        if match:
            return match.response_text
        return None

    def save_to_cache(
        self, query_text: str, query_embedding: list[float], response_text: str
    ) -> None:
        """
        Writes the query and its embedding to both PostgreSQL and Redis.

        Args:
            query_text: The original user query.
            query_embedding: The vector embedding of the query.
            response_text: The generated LLM response to cache.
        """
        # 1. Persist to PostgreSQL with pgvector
        cache_entry = SemanticCache(
            query_text=query_text,
            embedding=query_embedding,
            response_text=response_text,
        )
        self.db.add(cache_entry)
        self.db.commit()
        self.db.refresh(cache_entry)

        # 2. Persist to Redis
        if self.redis is not None:
            try:
                redis_key = f"semantic_cache:{query_text}"
                cache_payload = json.dumps({
                    "id": cache_entry.id,
                    "query_text": query_text,
                    "embedding": query_embedding,
                    "response_text": response_text,
                })
                self.redis.set(redis_key, cache_payload, ex=settings.CACHE_TTL_SECONDS)
            except Exception as e:
                logger.warning("Failed to save entry to Redis cache: %s", e)

    def get_exact_cached_response(self, query_text: str) -> str | None:
        """
        Queries Redis for an exact matching cached response (Tier 1 lookup).

        Args:
            query_text: The input query string to look up.

        Returns:
            Cached response text if found in Redis, otherwise None.
        """
        if self.redis is None:
            return None

        try:
            redis_key = f"semantic_cache:{query_text}"
            cached_data = self.redis.get(redis_key)
            if cached_data:
                parsed = json.loads(cached_data)
                return parsed.get("response_text")
        except Exception as e:
            logger.warning("Failed to retrieve entry from Redis cache: %s", e)

        return None
