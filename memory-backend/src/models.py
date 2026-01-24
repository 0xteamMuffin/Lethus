from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class MessageRequest(BaseModel):
    """Request for sending a message"""
    conversation_id: Optional[int] = None
    user_id: str
    message: str
    openai_api_key: str
    

class MessageResponse(BaseModel):
    """Response from the assistant"""
    conversation_id: int
    turn_id: int
    message: str
    retrieved_context: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ConversationCreate(BaseModel):
    """Create a new conversation"""
    user_id: str
    title: Optional[str] = "New Conversation"


class ConversationResponse(BaseModel):
    """Conversation details"""
    id: int
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TurnResponse(BaseModel):
    """Turn details"""
    id: int
    conversation_id: int
    turn_number: int
    user_message: str
    assistant_message: str
    timestamp: datetime
    importance_score: float
    is_pinned: bool
    
    class Config:
        from_attributes = True


class MemorySpan(BaseModel):
    """A span of conversation memory"""
    turn_ids: List[int]
    turns: List[Dict[str, Any]]
    relevance_score: float
    start_turn: int
    end_turn: int


class RetrievalResult(BaseModel):
    """Result from memory retrieval"""
    pinned_memories: List[Dict[str, Any]]
    relevant_spans: List[MemorySpan]
    total_tokens: int
    confidence_score: float
