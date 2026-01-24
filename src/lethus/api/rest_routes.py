"""
REST API routes for Lethus AI.
Handles conversations, messages, and user settings.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..storage.postgres import get_db, Conversation, Turn, User
from ..config import settings

router = APIRouter()


# === Pydantic Models ===

class UserSettingsRequest(BaseModel):
    user_id: str
    openai_api_key: Optional[str] = None


class UserSettingsResponse(BaseModel):
    user_id: str
    has_api_key: bool
    created_at: datetime
    updated_at: datetime


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


class TurnRequest(BaseModel):
    turn_number: int
    user_message: str
    assistant_message: str


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


@router.post("/conversations/{conversation_id}/turns", response_model=TurnResponse)
async def create_turn(
    conversation_id: int,
    request: TurnRequest,
    db: Session = Depends(get_db)
):
    """Create a new turn in a conversation"""
    # Verify conversation exists
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Create turn
    turn = Turn(
        conversation_id=conversation_id,
        turn_number=request.turn_number,
        user_message=request.user_message,
        assistant_message=request.assistant_message,
        timestamp=datetime.utcnow(),
        importance_score=0.0,
        is_pinned=False
    )
    
    db.add(turn)
    db.commit()
    db.refresh(turn)
    
    return TurnResponse(
        id=turn.id,
        conversation_id=turn.conversation_id,
        turn_number=turn.turn_number,
        user_message=turn.user_message,
        assistant_message=turn.assistant_message,
        timestamp=turn.timestamp,
        importance_score=turn.importance_score,
        is_pinned=turn.is_pinned
    )


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
    
    return {"message": "Conversation deleted successfully"}


# === Stats Endpoint ===

@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_conversations = db.query(Conversation).count()
    total_turns = db.query(Turn).count()
    total_users = db.query(User).count()
    
    return {
        "total_conversations": total_conversations,
        "total_turns": total_turns,
        "total_users": total_users,
        "version": "1.0.0"
    }
