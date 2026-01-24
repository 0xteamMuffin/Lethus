"""
PostgreSQL storage for conversation metadata and turns.
Used by the REST API for persistent conversation management.
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Generator

from ..config import settings

Base = declarative_base()


class User(Base):
    """Stores user settings and API keys"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True)  # Client-generated user ID
    openai_api_key = Column(String(500), nullable=True)  # Encrypted in production
    openai_base_url = Column(String(500), nullable=True)  # Custom base URL (None = use env default)
    # Model selections (None = use env defaults)
    llm_model = Column(String(255), nullable=True)  # e.g., "gpt-4o", "gpt-4o-mini"
    llm_temperature = Column(Float, nullable=True)  # e.g., 0.7
    llm_max_tokens = Column(Integer, nullable=True)  # e.g., 1000
    embedding_model = Column(String(255), nullable=True)  # e.g., "text-embedding-3-small"
    embedding_dim = Column(Integer, nullable=True)  # e.g., 1536
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    """Stores conversation metadata"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    title = Column(String(500))
    ghost_graph_json = Column(Text, default="{}")  # Serialized Ghost Graph
    enhanced_mode = Column(Boolean, default=True)  # True = use DYCP/Ghost Graph, False = normal passthrough
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Turn(Base):
    """Stores conversation turns (user message + assistant response)"""
    __tablename__ = "turns"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    turn_number = Column(Integer)
    user_message = Column(Text)
    assistant_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    importance_score = Column(Float, default=0.0)
    is_pinned = Column(Boolean, default=False)
    entities_json = Column(Text, default="[]")  # JSON list of entity names
    extra_data = Column(JSON, default={})


class PinnedMemory(Base):
    """Stores important/pinned memories"""
    __tablename__ = "pinned_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    turn_id = Column(Integer, index=True)
    content = Column(Text)
    importance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON, default={})


class PostgresStorage:
    """PostgreSQL storage manager"""
    
    def __init__(self, url: str = None):
        self.url = url or settings.postgres_url
        self.engine = create_engine(self.url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def init_db(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session (generator for dependency injection)"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_session(self) -> Session:
        """Create a new session directly"""
        return self.SessionLocal()


# Default instance
_postgres_storage = None


def get_postgres_storage() -> PostgresStorage:
    """Get or create the default PostgreSQL storage instance"""
    global _postgres_storage
    if _postgres_storage is None:
        _postgres_storage = PostgresStorage()
    return _postgres_storage


def init_db():
    """Initialize database tables"""
    storage = get_postgres_storage()
    storage.init_db()


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    storage = get_postgres_storage()
    yield from storage.get_session()
