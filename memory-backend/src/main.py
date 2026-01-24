from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from typing import List

from .config import settings
from .database import init_db, get_db, Conversation, Turn
from .milvus_client import milvus_client
from .models import (
    MessageRequest,
    MessageResponse,
    ConversationCreate,
    ConversationResponse,
    TurnResponse
)
from .memory_orchestrator import MemoryOrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    init_db()
    milvus_client.connect()
    yield
    # Shutdown
    milvus_client.disconnect()


app = FastAPI(
    title="DYCP Memory Backend",
    description="Conversational memory system with DYCP retrieval",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DYCP Memory Backend",
        "version": "0.1.0"
    }


@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation"""
    new_conversation = Conversation(
        user_id=conversation.user_id,
        title=conversation.title
    )
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    return new_conversation


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get conversation details"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation


@app.get("/api/conversations/user/{user_id}", response_model=List[ConversationResponse])
async def get_user_conversations(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get all conversations for a user"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).all()
    
    return conversations


@app.get("/api/conversations/{conversation_id}/turns", response_model=List[TurnResponse])
async def get_conversation_turns(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get all turns in a conversation"""
    turns = db.query(Turn).filter(
        Turn.conversation_id == conversation_id
    ).order_by(Turn.turn_number).all()
    
    return turns


@app.post("/api/chat", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message and get a response with memory-enhanced context
    
    This endpoint implements the complete DYCP memory flow:
    1. Importance detection → pinned memory
    2. Turn pairing
    3. Query embedding (+ variants)
    4. Weighted relevance scoring
    5. DYCP span selection
    6. Span merging & gap filling
    7. Token-aware trimming
    8. Confidence check & fallback
    9. Prompt assembly
    10. LLM response
    """
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        
        if not conversation_id:
            # Create new conversation
            conversation = Conversation(
                user_id=request.user_id,
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            conversation_id = conversation.id
        else:
            # Verify conversation exists
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
        
        # Initialize memory orchestrator
        orchestrator = MemoryOrchestrator(
            openai_api_key=request.openai_api_key,
            db=db
        )
        
        # Process message through memory pipeline
        result = await orchestrator.process_message(
            conversation_id=conversation_id,
            user_message=request.message,
            user_id=request.user_id
        )
        
        return MessageResponse(
            conversation_id=result["conversation_id"],
            turn_id=result["turn_id"],
            message=result["message"],
            retrieved_context=result["retrieved_context"],
            metadata=result["metadata"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its turns"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Delete all turns
    db.query(Turn).filter(Turn.conversation_id == conversation_id).delete()
    
    # Delete conversation
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
