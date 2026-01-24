from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Generator

from ..config import settings

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True)
    openai_api_key = Column(String(500), nullable=True)
    openai_base_url = Column(String(500), nullable=True)
    llm_model = Column(String(255), nullable=True)
    llm_temperature = Column(Float, nullable=True)
    llm_max_tokens = Column(Integer, nullable=True)
    embedding_model = Column(String(255), nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    title = Column(String(500))
    ghost_graph_json = Column(Text, default="{}")
    enhanced_mode = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Turn(Base):
    __tablename__ = "turns"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    turn_number = Column(Integer)
    user_message = Column(Text)
    assistant_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    importance_score = Column(Float, default=0.0)
    is_pinned = Column(Boolean, default=False)
    entities_json = Column(Text, default="[]")
    extra_data = Column(JSON, default={})


class PinnedMemory(Base):
    __tablename__ = "pinned_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    turn_id = Column(Integer, index=True)
    content = Column(Text)
    importance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON, default={})


class PostgresStorage:
    def __init__(self, url: str = None):
        self.url = url or settings.postgres_url
        self.engine = create_engine(self.url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def init_db(self):
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_session(self) -> Session:
        return self.SessionLocal()


_postgres_storage = None


def get_postgres_storage() -> PostgresStorage:
    global _postgres_storage
    if _postgres_storage is None:
        _postgres_storage = PostgresStorage()
    return _postgres_storage


def init_db():
    storage = get_postgres_storage()
    storage.init_db()


def get_db() -> Generator[Session, None, None]:
    storage = get_postgres_storage()
    yield from storage.get_session()
