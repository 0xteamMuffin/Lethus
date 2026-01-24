from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .config import settings

Base = declarative_base()


class Conversation(Base):
    """Stores conversation metadata"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    title = Column(String(500))
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
    metadata = Column(JSON, default={})


class PinnedMemory(Base):
    """Stores important/pinned memories"""
    __tablename__ = "pinned_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, index=True)
    turn_id = Column(Integer, index=True)
    content = Column(Text)
    importance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default={})


# Database engine and session
engine = create_engine(settings.postgres_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
