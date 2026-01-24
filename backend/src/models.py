"""Database models for PostgreSQL storage."""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Session(Base):
    """Conversation session."""
    
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    turns = relationship("Turn", back_populates="session", cascade="all, delete-orphan")


class Turn(Base):
    """Individual conversation turn (user message + assistant response)."""
    
    __tablename__ = "turns"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=True)
    importance_score = Column(Float, default=0.0)
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens_user = Column(Integer, default=0)
    tokens_assistant = Column(Integer, default=0)
    metadata = Column(JSON, default={})
    
    session = relationship("Session", back_populates="turns")


class PinnedMemory(Base):
    """Explicitly pinned important memories."""
    
    __tablename__ = "pinned_memories"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    turn_id = Column(String, ForeignKey("turns.id"), nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
