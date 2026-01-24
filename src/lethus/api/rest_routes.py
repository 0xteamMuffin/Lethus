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
    openai_base_url: Optional[str] = None  # Custom base URL (None = use env default)
    llm_model: Optional[str] = None  # Selected LLM model (None = use env default)
    llm_temperature: Optional[float] = None  # Temperature (None = use env default)
    llm_max_tokens: Optional[int] = None  # Max tokens (None = use env default)
    embedding_model: Optional[str] = None  # Selected embedding model (None = use env default)
    embedding_dim: Optional[int] = None  # Embedding dimension (None = use env default)


class UserSettingsResponse(BaseModel):
    user_id: str
    has_api_key: bool
    openai_base_url: Optional[str] = None  # User's custom base URL
    llm_model: Optional[str] = None  # User's selected LLM model
    llm_temperature: Optional[float] = None  # User's temperature setting
    llm_max_tokens: Optional[int] = None  # User's max tokens setting
    embedding_model: Optional[str] = None  # User's selected embedding model
    embedding_dim: Optional[int] = None  # User's embedding dimension
    # Environment defaults shown to frontend
    default_openai_base_url: str
    default_llm_model: str
    default_llm_temperature: float
    default_llm_max_tokens: int
    default_embedding_model: str
    default_embedding_dim: int
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
    base_url: Optional[str] = None  # Custom base URL (None = use env default)
    model: Optional[str] = None  # Model to test with (None = use env default)


class ValidateApiKeyResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class ConversationRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Conversation"
    enhanced_mode: Optional[bool] = True  # True = use DYCP/Ghost Graph


class ConversationResponse(BaseModel):
    id: int
    user_id: str
    title: str
    enhanced_mode: bool
    created_at: datetime
    updated_at: datetime


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    enhanced_mode: Optional[bool] = None


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
    
    # Update base URL if provided (empty string clears it to use default)
    if request.openai_base_url is not None:
        user.openai_base_url = request.openai_base_url if request.openai_base_url else None
        user.updated_at = datetime.utcnow()
    
    # Update LLM model if provided (empty string clears it to use default)
    if request.llm_model is not None:
        user.llm_model = request.llm_model if request.llm_model else None
        user.updated_at = datetime.utcnow()
    
    # Update temperature if provided (None clears it to use default)
    if request.llm_temperature is not None:
        user.llm_temperature = request.llm_temperature if request.llm_temperature >= 0 else None
        user.updated_at = datetime.utcnow()
    
    # Update max tokens if provided (None clears it to use default)
    if request.llm_max_tokens is not None:
        user.llm_max_tokens = request.llm_max_tokens if request.llm_max_tokens > 0 else None
        user.updated_at = datetime.utcnow()
    
    # Update embedding model if provided (empty string clears it to use default)
    if request.embedding_model is not None:
        user.embedding_model = request.embedding_model if request.embedding_model else None
        user.updated_at = datetime.utcnow()
    
    # Update embedding dimension if provided (None clears it to use default)
    if request.embedding_dim is not None:
        user.embedding_dim = request.embedding_dim if request.embedding_dim > 0 else None
        user.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(user)
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        openai_base_url=user.openai_base_url,
        llm_model=user.llm_model,
        llm_temperature=user.llm_temperature,
        llm_max_tokens=user.llm_max_tokens,
        embedding_model=user.embedding_model,
        embedding_dim=user.embedding_dim,
        default_openai_base_url=settings.openai_base_url,
        default_llm_model=settings.llm_model,
        default_llm_temperature=settings.llm_temperature,
        default_llm_max_tokens=settings.llm_max_tokens,
        default_embedding_model=settings.openai_embedding_model,
        default_embedding_dim=settings.openai_embedding_dim,
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
            openai_base_url=None,
            llm_model=None,
            llm_temperature=None,
            llm_max_tokens=None,
            embedding_model=None,
            embedding_dim=None,
            default_openai_base_url=settings.openai_base_url,
            default_llm_model=settings.llm_model,
            default_llm_temperature=settings.llm_temperature,
            default_llm_max_tokens=settings.llm_max_tokens,
            default_embedding_model=settings.openai_embedding_model,
            default_embedding_dim=settings.openai_embedding_dim,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    return UserSettingsResponse(
        user_id=user.user_id,
        has_api_key=bool(user.openai_api_key),
        openai_base_url=user.openai_base_url,
        llm_model=user.llm_model,
        llm_temperature=user.llm_temperature,
        llm_max_tokens=user.llm_max_tokens,
        embedding_model=user.embedding_model,
        embedding_dim=user.embedding_dim,
        default_openai_base_url=settings.openai_base_url,
        default_llm_model=settings.llm_model,
        default_llm_temperature=settings.llm_temperature,
        default_llm_max_tokens=settings.llm_max_tokens,
        default_embedding_model=settings.openai_embedding_model,
        default_embedding_dim=settings.openai_embedding_dim,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.post("/settings/validate-api-key", response_model=ValidateApiKeyResponse)
async def validate_api_key(request: ValidateApiKeyRequest):
    """Validate an API key by making a minimal chat completion request"""
    # Use provided values if set, otherwise use env defaults
    effective_base_url = request.base_url or settings.openai_base_url
    effective_base_url = effective_base_url.rstrip("/")
    effective_model = request.model or settings.llm_model
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{effective_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": effective_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1
                }
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
    """
    Try to fetch available models from the API.
    Falls back to empty lists if API doesn't support /models endpoint.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if not user or not user.openai_api_key:
        return AvailableModelsResponse(llm_models=[], embedding_models=[])
    
    # Use user's custom base_url if set, otherwise use env default
    effective_base_url = user.openai_base_url or settings.openai_base_url
    base_url = effective_base_url.rstrip("/")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/models",
                headers={
                    "Authorization": f"Bearer {user.openai_api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                models_data = data.get("data", [])
                
                llm_models = []
                embedding_models = []
                
                for model in models_data:
                    model_id = model.get("id", "")
                    # Categorize models based on common naming patterns
                    if "embed" in model_id.lower():
                        embedding_models.append(model_id)
                    else:
                        llm_models.append(model_id)
                
                return AvailableModelsResponse(
                    llm_models=sorted(llm_models),
                    embedding_models=sorted(embedding_models)
                )
            else:
                # API doesn't support /models - return empty lists
                return AvailableModelsResponse(llm_models=[], embedding_models=[])
                
    except Exception as e:
        # Any error - just return empty, user can type custom model names
        return AvailableModelsResponse(llm_models=[], embedding_models=[])


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
        title=request.title,
        enhanced_mode=request.enhanced_mode if request.enhanced_mode is not None else True
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        enhanced_mode=conversation.enhanced_mode,
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
            enhanced_mode=conv.enhanced_mode if conv.enhanced_mode is not None else True,
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
        enhanced_mode=conversation.enhanced_mode if conversation.enhanced_mode is not None else True,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: int, request: ConversationUpdateRequest, db: Session = Depends(get_db)):
    """Update a conversation (title or enhanced_mode). Note: enhanced_mode can only be turned ON, not OFF."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if request.title is not None:
        conversation.title = request.title
    
    # enhanced_mode can only be turned ON (True), once on it stays on
    if request.enhanced_mode is True and not conversation.enhanced_mode:
        conversation.enhanced_mode = True
    
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        enhanced_mode=conversation.enhanced_mode if conversation.enhanced_mode is not None else True,
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
