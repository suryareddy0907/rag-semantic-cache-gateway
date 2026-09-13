from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx
import openai
import redis
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, get_db
from app.services.cache_service import SemanticCacheService
from app.services.llm_service import LLMService, llm_service

logger = logging.getLogger(__name__)

router = APIRouter()

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a shared Redis client instance configured with fast socket timeouts (2s).
    If Redis is unreachable or times out, safely falls back to None so the gateway
    can continue operating with PostgreSQL pgvector.
    """
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None

    try:
        client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            retry_on_timeout=False,
        )
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:
        logger.warning("Redis is unavailable (%s). Continuing without Redis.", exc)
        return None


class QueryRequest(BaseModel):
    """
    Request model for semantic cache gateway query.
    """
    query: str = Field(
        ...,
        description="The prompt or query to be processed by the RAG Gateway.",
        min_length=1,
        examples=["What are the benefits of semantic caching in RAG?"],
    )


class QueryResponse(BaseModel):
    """
    Response model returned on a cache hit.
    """
    query: str
    response: str
    cached: bool
    source: Optional[str] = None


def save_cache_task(
    query: str,
    embedding: list[float],
    response_chunks: list[str],
) -> None:
    """
    Background task to compile the streamed response chunks and persist
    the query, vector embedding, and full response to both PostgreSQL (pgvector)
    and Redis.
    """
    full_response = "".join(response_chunks).strip()
    if not full_response:
        logger.warning("Empty response received; skipping cache save for: %s", query)
        return

    db = SessionLocal()
    try:
        redis_client = get_redis_client()
        cache_service = SemanticCacheService(db=db, redis_client=redis_client)
        cache_service.save_to_cache(
            query_text=query,
            query_embedding=embedding,
            response_text=full_response,
        )
        logger.info("Asynchronously saved query and response to semantic cache.")
    except Exception as exc:
        logger.error("Failed to save to semantic cache in background: %s", exc)
    finally:
        db.close()


@router.post(
    "/query",
    summary="Query RAG Semantic Cache Gateway",
    response_description="Returns cached response immediately or streams chunks via SSE",
)
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    llm: LLMService = Depends(lambda: llm_service),
):
    """
    Query processing pipeline with two-tier semantic caching, explicit timeouts,
    and fast 504 HTTP error handling:
    1. Generate the query embedding using LLMService (with timeout).
    2. Check Redis (Tier 1) and PostgreSQL pgvector (Tier 2) via SemanticCacheService.
    3. If cached response found, return immediately with low latency.
    4. If cache miss, initiate streaming completion from LLM with strict timeouts.
    5. Asynchronously persist compiled response and embedding in background.
    """
    # 1. Generate query embedding with explicit timeout handling
    try:
        embedding = await llm.get_embedding(request.query)
    except (openai.APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError) as exc:
        logger.error("Timeout generating embedding for query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Upstream embedding service timed out. Please try again.",
        )
    except openai.RateLimitError as exc:
        logger.error("Rate limit hit during embedding generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upstream embedding rate limit exceeded. Please throttle requests.",
        )
    except (openai.APIConnectionError, httpx.NetworkError) as exc:
        logger.error("Connection failure during embedding generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to connect to upstream embedding service.",
        )
    except Exception as exc:
        logger.error("Unexpected error during embedding generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation error: {str(exc)}",
        )

    # 2. Check Redis (Tier 1) and PostgreSQL pgvector (Tier 2)
    redis_client = get_redis_client()
    cache_service = SemanticCacheService(db=db, redis_client=redis_client)

    # Tier 1: Exact match lookup in Redis
    cached_response = cache_service.get_exact_cached_response(request.query)
    cache_source = "redis"

    # Tier 2: Cosine similarity vector search in PostgreSQL pgvector
    if not cached_response:
        try:
            cached_response = cache_service.get_similar_cached_response(
                query_embedding=embedding,
                threshold=settings.SIMILARITY_THRESHOLD,
            )
            cache_source = "pgvector"
        except (OperationalError, SQLAlchemyError, TimeoutError, Exception) as exc:
            # If the database connection pool is saturated or query times out,
            # fail open to LLM generation rather than hanging the user request
            logger.warning("Database cache search timed out or failed (%s). Falling back to LLM.", exc)
            cached_response = None

    # 3. Cache HIT: return immediately
    if cached_response:
        logger.info("Cache HIT from [%s] for query: %s", cache_source, request.query)
        return JSONResponse(
            content={
                "query": request.query,
                "response": cached_response,
                "cached": True,
                "source": cache_source,
            },
            headers={
                "X-Cache-Hit": "true",
                "X-Cache-Source": cache_source,
            },
        )

    # 4. Cache MISS: initiate streaming chat completion with timeout guards
    logger.info("Cache MISS for query: %s. Initiating LLM stream...", request.query)
    try:
        stream = await llm.create_chat_stream(request.query)
    except (openai.APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError) as exc:
        logger.error("LLM chat stream initiation timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Upstream LLM service timed out. Please try again.",
        )
    except openai.RateLimitError as exc:
        logger.error("LLM rate limit reached: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upstream LLM rate limit exceeded. Please try again shortly.",
        )
    except (openai.APIConnectionError, httpx.NetworkError) as exc:
        logger.error("Connection error reaching LLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to connect to upstream LLM service.",
        )
    except Exception as exc:
        logger.error("Failed to initiate LLM streaming: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to initiate LLM streaming: {str(exc)}",
        )

    collected_chunks: list[str] = []

    async def sse_stream_generator():
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_piece = chunk.choices[0].delta.content
                    collected_chunks.append(text_piece)
                    payload = json.dumps({"chunk": text_piece, "cached": False})
                    yield f"data: {payload}\n\n"
            # Stream completion event
            yield f"data: {json.dumps({'done': True})}\n\n"
        except (openai.APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError) as exc:
            logger.error("Timeout reading chunk from LLM stream: %s", exc)
            yield f"data: {json.dumps({'error': 'Upstream LLM stream timed out', 'status_code': 504})}\n\n"
        except Exception as exc:
            logger.error("Error during streaming completion: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    # 5. Save compiled response and embedding in background
    background_tasks.add_task(
        save_cache_task,
        query=request.query,
        embedding=embedding,
        response_chunks=collected_chunks,
    )

    return StreamingResponse(
        sse_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Cache-Hit": "false",
            "X-Accel-Buffering": "no",
        },
        background=background_tasks,
    )
