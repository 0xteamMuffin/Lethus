"""
REST API routes for Lethus AI.
Handles conversations, messages, and user settings.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx

from ..storage.postgres import get_db, Conversation, Turn, User
from ..config import settings

router = APIRouter()


# === Pydantic Models ===

class UserSettingsRequest(BaseModel):
    user_id: str
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = None  # Selected LLM model (None = use env default)
    embedding_model: Optional[str] = None  # Selected embedding model (None = use env default)


class UserSettingsResponse(BaseModel):
    user_id: str
    has_api_key: bool
    llm_model: Optional[str] = None  # User's selected LLM model
    embedding_model: Optional[str] = None  # User's selected embedding model
    default_llm_model: str  # Env default LLM model
    default_embedding_model: str  # Env default embedding model
    created_at: datetime
    updated_at: datetime


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "openai"
    created: Optional[int] = None


class AvailableModelsResponse(BaseModel):
    llm_models: List[ModelInfo]
    embedding_models: List[ModelInfo]


class ValidateApiKeyRequest(BaseModel):
    api_key: str


class ValidateApiKeyResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


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
    """Save or update user settings including API key and model preferences"""
    # Find or create user
    user = db.query(User).filter(User.user_id == request.user_id).first()
    
    if user is None:
        user = User(user_id=request.user_id)
        db.add(user)
    
    # Update API key if provided
    if request.openai_api_key is not None:
        user.openai_api_key = request.openai_api_key
        user.updated_at = datetime.utcnow()
    
    # Update LLM model if provided (empty string clears it to use default)
    if request.llm_model is not None:
        user.llm_model = request.llm_model if request.llm_model else None
        user.updated_at = datetime.utcnow()
    
    # Update embedding model if provided (empty string clears it to use default)
    if request.embedding_model is not None:
        user.embedding_model = request.embedding_model if request.embedding_model else None
        user.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(user)
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        llm_model=user.llm_model,
        embedding_model=user.embedding_model,
        default_llm_model=settings.llm_model,
        default_embedding_model=settings.openai_embedding_model,
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
            llm_model=None,
            embedding_model=None,
            default_llm_model=settings.llm_model,
            default_embedding_model=settings.openai_embedding_model,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        llm_model=user.llm_model,
        embedding_model=user.embedding_model,
        default_llm_model=settings.llm_model,
        default_embedding_model=settings.openai_embedding_model,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.post("/settings/validate-api-key", response_model=ValidateApiKeyResponse)
async def validate_api_key(request: ValidateApiKeyRequest):
    """Validate an OpenAI API key by making a test request"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.openai_base_url}/models",
                headers={"Authorization": f"Bearer {request.api_key}"}
            )
            
            if response.status_code == 200:
                return ValidateApiKeyResponse(valid=True)
            elif response.status_code == 401:
                return ValidateApiKeyResponse(valid=False, error="Invalid API key")
            else:
                return ValidateApiKeyResponse(
                    valid=False, 
                    error=f"API error: {response.status_code}"
                )
    except httpx.TimeoutException:
        return ValidateApiKeyResponse(valid=False, error="Request timed out")
    except Exception as e:
        return ValidateApiKeyResponse(valid=False, error=str(e))


@router.get("/settings/{user_id}/models", response_model=AvailableModelsResponse)
async def get_available_models(user_id: str, db: Session = Depends(get_db)):
    """Fetch available models from OpenAI using the user's API key"""
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if not user or not user.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Please set your OpenAI API key first."
        )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.openai_base_url}/models",
                headers={"Authorization": f"Bearer {user.openai_api_key}"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch models: {response.text}"
                )
            
            data = response.json()
            models = data.get("data", [])
            
            # Categorize models
            llm_models = []
            embedding_models = []
            
            # Known model patterns
            llm_patterns = ["gpt-", "o1-", "o3-", "chatgpt-"]
            embedding_patterns = ["embedding", "embed"]
            
            for model in models:
                model_id = model.get("id", "")
                model_info = ModelInfo(
                    id=model_id,
                    object=model.get("object", "model"),
                    owned_by=model.get("owned_by", "openai"),
                    created=model.get("created")
                )
                
                # Check if it's an embedding model
                if any(pattern in model_id.lower() for pattern in embedding_patterns):
                    embedding_models.append(model_info)
                # Check if it's an LLM model
                elif any(pattern in model_id.lower() for pattern in llm_patterns):
                    llm_models.append(model_info)
            
            # Sort models by ID for consistency
            llm_models.sort(key=lambda x: x.id)
            embedding_models.sort(key=lambda x: x.id)
            
            return AvailableModelsResponse(
                llm_models=llm_models,
                embedding_models=embedding_models
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to OpenAI timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/settings/{user_id}/api-key")
async def delete_api_key(user_id: str, db: Session = Depends(get_db)):
    """Remove the user's API key"""
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.openai_api_key = None
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "API key removed successfully"}


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
