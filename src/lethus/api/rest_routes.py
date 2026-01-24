"""
REST API routes for Lethus AI.
Handles conversations, messages, and user settings.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import numpy as np

from ..storage.postgres import get_db, Conversation, Turn, User
from ..core.dycp import DYCPCore
from ..core.ghost_graph import GhostGraph
from ..core.embeddings import get_embedding_provider
from ..storage.milvus import get_milvus_storage
from ..config import settings
import httpx

router = APIRouter()

# Global components
_dycp_core = None
_ghost_graphs = {}  # conversation_id -> GhostGraph


def get_dycp_core():
    """Get or create DYCP core instance"""
    global _dycp_core
    if _dycp_core is None:
        _dycp_core = DYCPCore()
    return _dycp_core


def get_ghost_graph(conversation_id: int) -> GhostGraph:
    """Get or create Ghost Graph for a conversation"""
    global _ghost_graphs
    if conversation_id not in _ghost_graphs:
        _ghost_graphs[conversation_id] = GhostGraph()
    return _ghost_graphs[conversation_id]


# === Pydantic Models ===

class UserSettingsRequest(BaseModel):
    user_id: str
    openai_api_key: Optional[str] = None


class UserSettingsResponse(BaseModel):
    user_id: str
    has_api_key: bool
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[int] = None
    openai_api_key: Optional[str] = None  # Deprecated - for backward compatibility


class ChatResponse(BaseModel):
    conversation_id: int
    turn_id: int
    message: str
    retrieved_context: Dict[str, Any]
    metadata: Dict[str, Any]


class ConversationRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Conversation"


class ConversationResponse(BaseModel):
    id: int
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class TurnResponse(BaseModel):
    id: int
    conversation_id: int
    turn_number: int
    user_message: str
    assistant_message: str
    timestamp: datetime
    importance_score: float
    is_pinned: bool


# === User Settings Endpoints ===

@router.post("/settings", response_model=UserSettingsResponse)
async def save_user_settings(request: UserSettingsRequest, db: Session = Depends(get_db)):
    """Save or update user settings including API key"""
    # Find or create user
    user = db.query(User).filter(User.user_id == request.user_id).first()
    
    if user is None:
        user = User(user_id=request.user_id)
        db.add(user)
    
    # Update API key if provided
    if request.openai_api_key is not None:
        user.openai_api_key = request.openai_api_key
        user.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(user)
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.get("/settings/{user_id}", response_model=UserSettingsResponse)
async def get_user_settings(user_id: str, db: Session = Depends(get_db)):
    """Get user settings (without exposing the API key itself)"""
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if user is None:
        # Return default settings indicating no API key
        return UserSettingsResponse(
            user_id=user_id,
            has_api_key=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        created_at=user.created_at,
        updated_at=user.updated_at
    )


# === Chat Endpoint ===

@router.post("/chat", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and get AI response with DYCP context"""
    
    # Get user's API key from database
    user = db.query(User).filter(User.user_id == request.user_id).first()
    
    # Use provided API key (for backward compatibility) or user's stored key
    api_key = request.openai_api_key or (user.openai_api_key if user else None)
    
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Please set your OpenAI API key in settings."
        )
    
    # Create or get conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == request.user_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=request.user_id,
            title=request.message[:50] + ("..." if len(request.message) > 50 else "")
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    # Get conversation history
    turns = db.query(Turn).filter(
        Turn.conversation_id == conversation.id
    ).order_by(Turn.turn_number).all()
    
    # Build message history for DYCP
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.user_message})
        messages.append({"role": "assistant", "content": turn.assistant_message})
    
    # Add current user message
    messages.append({"role": "user", "content": request.message})
    
    # Apply DYCP if we have history
    retrieved_context = {
        "spans": [],
        "pinned_memories": [],
        "confidence": {"confident": True, "score": 1.0}
    }
    
    if len(turns) > 0:
        dycp = get_dycp_core()
        ghost_graph = get_ghost_graph(conversation.id)
        embeddings = get_embedding_provider(api_key=api_key)
        
        # Get embeddings for history (excluding current message)
        history = messages[:-1]
        if history:
            history_texts = [m["content"] for m in history]
            history_embs = np.array([embeddings.embed_text(t) for t in history_texts])
            query_emb = embeddings.embed_text(request.message)
            
            # Compute relevance with decay
            turn_indices = np.arange(len(history))
            current_turn = len(history)
            
            similarities = dycp.compute_relevance(
                query_emb,
                history_embs,
                turn_indices=turn_indices,
                current_turn=current_turn,
                apply_decay=True
            )
            
            # Ghost Graph boost
            history_with_entities = []
            for i, m in enumerate(history):
                entities = ghost_graph.extract_entities(m["content"])
                history_with_entities.append({
                    "role": m["role"],
                    "content": m["content"],
                    "turn_index": i,
                    "entities": json.dumps([e["name"] for e in entities])
                })
            
            similarities = ghost_graph.boost_similarities(
                request.message,
                history_with_entities,
                similarities
            )
            
            # Get spans
            spans = dycp.get_pruned_indices(similarities)
            
            retrieved_context["spans"] = [
                {
                    "start_index": start,
                    "end_index": end,
                    "turns": [
                        {"role": history[i]["role"], "content": history[i]["content"]}
                        for i in range(start, min(end + 1, len(history)))
                    ],
                    "relevance_score": float(np.mean(similarities[start:end+1]))
                }
                for start, end in spans
            ]
            
            # Build reduced context for LLM
            reduced_messages = [{"role": "system", "content": "You are a helpful assistant."}]
            if spans:
                selected_indices = set()
                for start, end in spans:
                    for i in range(start, end + 1):
                        selected_indices.add(i)
                
                for i, m in enumerate(history):
                    if i in selected_indices:
                        reduced_messages.append(m)
            
            reduced_messages.append({"role": "user", "content": request.message})
            messages = reduced_messages
    else:
        # No history, just use system prompt + current message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.message}
        ]
    
    # Call OpenAI API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "temperature": 0.7
                }
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenAI API error: {error_detail}"
                )
            
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI response: {str(e)}")
    
    # Save turn to database
    turn_number = len(turns) + 1
    turn = Turn(
        conversation_id=conversation.id,
        turn_number=turn_number,
        user_message=request.message,
        assistant_message=assistant_message,
        timestamp=datetime.utcnow(),
        importance_score=0.0,
        is_pinned=False
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    
    # Update Ghost Graph
    ghost_graph = get_ghost_graph(conversation.id)
    user_entities = ghost_graph.extract_entities(request.message)
    assistant_entities = ghost_graph.extract_entities(assistant_message)
    
    return ChatResponse(
        conversation_id=conversation.id,
        turn_id=turn.id,
        message=assistant_message,
        retrieved_context=retrieved_context,
        metadata={
            "turn_number": turn_number,
            "entities": [e["name"] for e in user_entities + assistant_entities]
        }
    )


# === Conversation Endpoints ===

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest, db: Session = Depends(get_db)):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=request.user_id,
        title=request.title
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("/conversations/user/{user_id}", response_model=List[ConversationResponse])
async def get_user_conversations(user_id: str, db: Session = Depends(get_db)):
    """Get all conversations for a user"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).all()
    
    return [
        ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at
        )
        for conv in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get a specific conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("/conversations/{conversation_id}/turns", response_model=List[TurnResponse])
async def get_conversation_turns(conversation_id: int, db: Session = Depends(get_db)):
    """Get all turns for a conversation"""
    turns = db.query(Turn).filter(
        Turn.conversation_id == conversation_id
    ).order_by(Turn.turn_number).all()
    
    return [
        TurnResponse(
            id=turn.id,
            conversation_id=turn.conversation_id,
            turn_number=turn.turn_number,
            user_message=turn.user_message,
            assistant_message=turn.assistant_message,
            timestamp=turn.timestamp,
            importance_score=turn.importance_score,
            is_pinned=turn.is_pinned
        )
        for turn in turns
    ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete a conversation and all its turns"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete all turns
    db.query(Turn).filter(Turn.conversation_id == conversation_id).delete()
    
    # Delete conversation
    db.delete(conversation)
    db.commit()
    
    # Clean up ghost graph
    global _ghost_graphs
    if conversation_id in _ghost_graphs:
        del _ghost_graphs[conversation_id]
    
    return {"message": "Conversation deleted successfully"}


# === Stats Endpoint ===

@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_conversations = db.query(Conversation).count()
    total_turns = db.query(Turn).count()
    total_users = db.query(User).count()
    
    dycp = get_dycp_core()
    
    return {
        "total_conversations": total_conversations,
        "total_turns": total_turns,
        "total_users": total_users,
        "decay_lambda": dycp.decay_lambda,
        "relevance_threshold": dycp.relevance_threshold,
        "version": "1.0.0"
    }
