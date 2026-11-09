"""
Database Connection and Session Management.
Supports PostgreSQL with fallback for testing/local setups.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
