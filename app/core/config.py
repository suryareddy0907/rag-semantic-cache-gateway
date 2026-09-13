from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables
    or a .env file, with fallback defaults for local development.
    """
    # Core Application Settings
    PROJECT_NAME: str = "RAG Semantic Cache Gateway"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database Settings (PostgreSQL + pgvector)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/rag_gateway"

    # Redis Cache Settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI & LLM Configuration
    OPENAI_API_KEY: str = "mock-openai-key-for-local-dev"
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Semantic Cache Configuration
    SIMILARITY_THRESHOLD: float = 0.85
    CACHE_TTL_SECONDS: int = 3600

    # Mock Provider Configuration (for local load testing)
    USE_MOCK_EMBEDDINGS: bool = False
    EMBEDDING_DIMENSIONS: int = 3072

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance, avoiding repeated disk reads
    and environment parsing on every call or dependency injection.
    """
    return Settings()


settings: Settings = get_settings()
