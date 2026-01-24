"""
Lethus REST API.
FastAPI endpoints for web application integration.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from typing import List
import numpy as np
import json

from ..config import settings
from ..storage.postgres import init_db, get_db, Conversation, Turn, PinnedMemory
from ..storage.milvus import get_milvus_storage
from ..core.dycp import DYCPCore
from ..core.ghost_graph import GhostGraph
from ..core.prefetch import PrefetchCache, Prefetcher
from ..core.embeddings import get_embedding_provider
from .models import (
    MessageRequest,
    MessageResponse,
    ConversationCreate,
    ConversationResponse,
    TurnResponse,
    MemoryStats
)

# Global components
_components = {}


def _get_components(api_key: str = None):
    """Get or initialize components"""
    if "dycp" not in _components:
        _components["dycp"] = DYCPCore()
        _components["ghost_graphs"] = {}  # Per-conversation ghost graphs
        _components["prefetch_cache"] = PrefetchCache()
        _components["milvus"] = get_milvus_storage()
    
    # Get embedding provider (may need API key)
    key = "embeddings" if api_key is None else f"embeddings_{api_key[:8]}"
    if key not in _components:
        _components[key] = get_embedding_provider(api_key=api_key)
    
    return _components


def _get_ghost_graph(conversation_id: int, db: Session) -> GhostGraph:
    """Get or create Ghost Graph for a conversation"""
    components = _get_components()
    
    if conversation_id not in components["ghost_graphs"]:
        graph = GhostGraph()
        
        # Load from database if exists
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.ghost_graph_json:
            try:
                graph.from_json(conv.ghost_graph_json)
            except:
                pass
        
        components["ghost_graphs"][conversation_id] = graph
    
    return components["ghost_graphs"][conversation_id]


def _save_ghost_graph(conversation_id: int, db: Session):
    """Save Ghost Graph to database"""
    components = _get_components()
    
    if conversation_id in components["ghost_graphs"]:
        graph = components["ghost_graphs"][conversation_id]
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.ghost_graph_json = graph.to_json()
            db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    init_db()
    yield


app = FastAPI(
    title="Lethus API",
    description="DYCP Memory System with Kadane's Algorithm, Semantic Decay, and Ghost Graph",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Lethus API",
        "version": "0.1.0"
    }


@app.get("/api/stats", response_model=MemoryStats)
async def get_stats(db: Session = Depends(get_db)):
    """Get memory system statistics"""
    components = _get_components()
    milvus = components["milvus"]
    
    total_turns = milvus.get_turn_count()
    total_entities = sum(len(g.entities) for g in components.get("ghost_graphs", {}).values())
    
    return MemoryStats(
        total_turns=total_turns,
        tracked_entities=total_entities,
        cache_size=len(components["prefetch_cache"]),
        decay_lambda=settings.decay_lambda
    )


# === Conversation Endpoints ===

@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation"""
    new_conv = Conversation(
        user_id=conversation.user_id,
        title=conversation.title
    )
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return new_conv


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get conversation details"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/api/conversations/user/{user_id}", response_model=List[ConversationResponse])
async def get_user_conversations(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get all conversations for a user"""
    convs = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).all()
    return convs


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


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its data"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete turns and pinned memories
    db.query(Turn).filter(Turn.conversation_id == conversation_id).delete()
    db.query(PinnedMemory).filter(PinnedMemory.conversation_id == conversation_id).delete()
    db.delete(conv)
    db.commit()
    
    # Clear from Milvus
    components = _get_components()
    components["milvus"].clear_conversation(conversation_id)
    
    # Clear ghost graph
    if conversation_id in components.get("ghost_graphs", {}):
        del components["ghost_graphs"][conversation_id]
    
    return {"message": "Conversation deleted"}


# === Chat Endpoint ===

@app.post("/api/chat", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a message and get a memory-enhanced response.
    
    Implements the DYCP memory flow:
    1. Create/get conversation
    2. Retrieve context using DYCP (Kadane's Algorithm)
    3. Apply semantic decay and ghost graph boosting
    4. Generate response with LLM
    5. Store turn and extract entities
    """
    components = _get_components(request.openai_api_key)
    
    # Get or create conversation
    if request.conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.id == request.conversation_id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = conv.id
    else:
        conv = Conversation(
            user_id=request.user_id,
            title=request.message[:50] + "..." if len(request.message) > 50 else request.message
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id
    
    # Get turn number
    last_turn = db.query(Turn).filter(
        Turn.conversation_id == conversation_id
    ).order_by(Turn.turn_number.desc()).first()
    turn_number = (last_turn.turn_number + 1) if last_turn else 1
    
    # Get components
    dycp = components["dycp"]
    milvus = components["milvus"]
    embeddings = components.get(f"embeddings_{request.openai_api_key[:8]}" if request.openai_api_key else "embeddings")
    if not embeddings:
        embeddings = get_embedding_provider(api_key=request.openai_api_key)
    
    ghost_graph = _get_ghost_graph(conversation_id, db)
    
    # Retrieve context using DYCP
    turns, history_embs = milvus.get_all_turns_ordered(conversation_id)
    
    retrieved_spans = []
    confidence = {"confident": True, "score": 1.0}
    
    if turns:
        query_emb = embeddings.embed_text(request.message)
        turn_indices = np.array([t["turn_index"] for t in turns])
        
        # Compute relevance with decay
        similarities = dycp.compute_relevance(
            query_emb,
            history_embs,
            turn_indices=turn_indices,
            current_turn=len(turns),
            apply_decay=True
        )
        
        # Ghost Graph boost
        similarities = ghost_graph.boost_similarities(request.message, turns, similarities)
        
        # Get spans
        spans = dycp.get_pruned_indices(similarities)
        
        for start, end in spans:
            span_turns = [turns[i] for i in range(start, end + 1)]
            avg_sim = np.mean(similarities[start:end + 1])
            retrieved_spans.append({
                "start_index": start,
                "end_index": end,
                "turns": span_turns,
                "relevance_score": float(avg_sim)
            })
        
        # Calculate confidence
        if spans:
            max_sim = max(similarities)
            confidence = {
                "confident": bool(max_sim > settings.confidence_threshold),
                "score": float(max_sim)
            }
    
    # Get pinned memories
    pinned = db.query(PinnedMemory).filter(
        PinnedMemory.conversation_id == conversation_id
    ).order_by(PinnedMemory.importance_score.desc()).limit(5).all()
    
    pinned_memories = [
        {"content": p.content, "importance_score": p.importance_score}
        for p in pinned
    ]
    
    # Build prompt and call LLM
    from openai import OpenAI
    
    api_key = request.openai_api_key or settings.openai_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")
    
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with access to conversation history."}
    ]
    
    # Add pinned memories
    if pinned_memories:
        pinned_text = "Important information:\n" + "\n".join(
            f"- {m['content']}" for m in pinned_memories
        )
        messages.append({"role": "system", "content": pinned_text})
    
    # Add retrieved context
    if retrieved_spans:
        context_text = "Relevant conversation history:\n"
        for span in retrieved_spans:
            for t in span["turns"]:
                context_text += f"[{t['role'].upper()}]: {t['content']}\n"
        messages.append({"role": "system", "content": context_text})
    
    messages.append({"role": "user", "content": request.message})
    
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens
    )
    
    assistant_message = response.choices[0].message.content
    
    # Extract entities and store turn
    user_entities = ghost_graph.extract_entities(request.message)
    assistant_entities = ghost_graph.extract_entities(assistant_message)
    all_entities = user_entities + assistant_entities
    entity_names = [e["name"] for e in all_entities]
    
    ghost_graph.register_entities(all_entities)
    _save_ghost_graph(conversation_id, db)
    
    # Store in database
    turn = Turn(
        conversation_id=conversation_id,
        turn_number=turn_number,
        user_message=request.message,
        assistant_message=assistant_message,
        entities_json=json.dumps(entity_names)
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    
    # Store in Milvus
    combined_text = f"User: {request.message}\nAssistant: {assistant_message}"
    embedding = embeddings.embed_text(combined_text)
    milvus.add_turn(
        role="combined",
        content=combined_text,
        embedding=embedding,
        conversation_id=conversation_id,
        entities=json.dumps(entity_names)
    )
    
    return MessageResponse(
        conversation_id=conversation_id,
        turn_id=turn.id,
        message=assistant_message,
        retrieved_context={
            "spans": retrieved_spans,
            "pinned_memories": pinned_memories,
            "confidence": confidence
        },
        metadata={
            "turn_number": turn_number,
            "entities": entity_names
        }
    )


def main():
    """Entry point for REST API server"""
    import uvicorn
    uvicorn.run(
        "lethus.api.rest:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
