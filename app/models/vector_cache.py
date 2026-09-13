from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, func
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class SemanticCache(Base):
    """
    SQLAlchemy model for persisting queries, their vector embeddings,
    and LLM responses for semantic similarity caching with pgvector.
    """
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_text = Column(String, nullable=False, index=True)
    embedding = Column(Vector(3072), nullable=False)
    response_text = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<SemanticCache(id={self.id}, query_text='{self.query_text[:30]}...')>"
