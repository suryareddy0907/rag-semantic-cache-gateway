import logging
import random
import uuid
from locust import HttpUser, between, events, task

logger = logging.getLogger(__name__)

# Fixed set of popular recurring queries to benchmark Tier 1 (Redis) exact cache hits
POPULAR_QUERIES = [
    "What is semantic caching in RAG pipelines?",
    "How does pgvector index high-dimensional embeddings?",
    "What is the difference between exact cache and semantic cache?",
    "How does Redis improve latency in an LLM gateway?",
    "What is cosine similarity and cosine distance in vector search?",
]

# Semantic variants of the popular queries to benchmark Tier 2 (pgvector) similarity hits
SEMANTIC_VARIANTS = [
    "Explain semantic caching in retrieval-augmented generation architectures.",
    "How are vector embeddings indexed and searched in PostgreSQL pgvector?",
    "Compare exact key-value caching with vector similarity caching.",
    "What are the latency benefits of using Redis for caching LLM responses?",
    "Can you explain the mathematical difference between cosine similarity and distance?",
]

# Novel query templates to benchmark cache misses and streaming LLM completions
COLD_QUERY_TEMPLATES = [
    "Generate a unique technical explanation for topic identifier {uid}",
    "What were the major developments in computing during the year {year}?",
    "Write a short technical summary about scenario code {uid}",
    "Provide three distinct engineering tradeoffs for system module {uid}",
]


class RAGGatewayUser(HttpUser):
    """
    Simulated user performing concurrent requests against the RAG Semantic Cache Gateway.
    Benchmarks the latency and throughput differences between:
    - Exact cache hits (Tier 1 Redis)
    - Semantic similarity cache hits (Tier 2 pgvector)
    - Full cache misses (Streaming LLM completions)
    """

    # Wait between 100ms and 500ms between requests to generate realistic concurrency
    wait_time = between(2, 5)

    def _send_query(self, query: str, scenario: str) -> None:
        """
        Sends a POST request to /api/v1/query and records stats under a scenario-specific name.
        """
        endpoint_name = f"/api/v1/query [{scenario}]"
        payload = {"query": query}

        with self.client.post(
            "/api/v1/query",
            json=payload,
            name=endpoint_name,
            catch_response=True,
            timeout=30.0,
        ) as response:
            if response.status_code == 200:
                is_cached = response.headers.get("X-Cache-Hit") == "true"
                cache_source = response.headers.get("X-Cache-Source", "miss")
                content_type = response.headers.get("Content-Type", "")

                # Verify streaming response vs JSON response
                if "text/event-stream" in content_type:
                    # Cache miss: consume streamed SSE data
                    response.success()
                else:
                    # Cache hit: parse immediate JSON
                    try:
                        data = response.json()
                        if data.get("cached"):
                            response.success()
                        else:
                            response.success()
                    except Exception as e:
                        response.failure(f"Failed to parse JSON response: {e}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(6)
    def test_popular_queries_exact_hit(self) -> None:
        """
        Sends identical recurring queries (60% weight).
        Expects Tier 1 (Redis) or Tier 2 (pgvector) low-latency cache hits.
        """
        query = random.choice(POPULAR_QUERIES)
        self._send_query(query, scenario="exact_hit")

    @task(3)
    def test_semantic_variant_hit(self) -> None:
        """
        Sends semantic rephrasings of popular queries (30% weight).
        Expects Tier 2 (pgvector) cosine similarity cache hits.
        """
        query = random.choice(SEMANTIC_VARIANTS)
        self._send_query(query, scenario="semantic_hit")

    @task(1)
    def test_cold_query_cache_miss(self) -> None:
        """
        Sends unique / novel queries (10% weight).
        Expects cache misses triggering LLM generation and SSE streaming.
        """
        template = random.choice(COLD_QUERY_TEMPLATES)
        query = template.format(
            uid=uuid.uuid4().hex[:8],
            year=random.randint(1960, 2024),
        )
        self._send_query(query, scenario="cache_miss")


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    logger.info("Starting RAG Semantic Cache Gateway Locust Benchmark...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    logger.info("RAG Semantic Cache Gateway Locust Benchmark completed.")
