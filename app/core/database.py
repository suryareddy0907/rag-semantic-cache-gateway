from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings

# Configure connection arguments (e.g. for sqlite if used in testing)
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Session factory for generating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative Base class for SQLAlchemy models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request
    and ensures it is cleanly closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
