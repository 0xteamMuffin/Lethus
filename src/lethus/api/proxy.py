"""
Lethus OpenAI-Compatible Proxy.
Drop-in replacement for OpenAI API that applies DYCP context reduction.

Usage:
    Set base_url="http://localhost:8000/v1" in any OpenAI-compatible client.
    Your API key is passed through to the real LLM provider.
"""
import json
import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, AsyncIterator
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np

from ..config import settings
from ..storage.milvus import get_milvus_storage
from ..storage.postgres import get_db, User, Conversation
from ..core.dycp import DYCPCore
from ..core.ghost_graph import GhostGraph
from ..core.embeddings import get_embedding_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("lethus.dycp")


@dataclass
class DYCPStats:
    """Statistics from DYCP context reduction."""
    original_messages: int = 0
    reduced_messages: int = 0
    original_tokens: int = 0
    reduced_tokens: int = 0
    spans_found: int = 0
    span_details: List[tuple] = field(default_factory=list)
    ghost_graph_entities: int = 0
    ghost_graph_boosts: int = 0
    decay_lambda: float = 0.0
    processing_time_ms: float = 0.0
    # Extended dev info
    enhanced_mode: bool = True
    entity_names: List[str] = field(default_factory=list)  # List of detected entities
    boost_details: List[dict] = field(default_factory=list)  # Entity boost breakdown
    tau: float = 0.0  # Gain threshold used
    theta: float = 0.0  # Stopping threshold used
    similarity_scores: List[float] = field(default_factory=list)  # Per-turn similarity scores
    
    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.reduced_tokens
    
    @property
    def reduction_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100
    
    @property
    def message_reduction_percent(self) -> float:
        if self.original_messages == 0:
            return 0.0
        return ((self.original_messages - self.reduced_messages) / self.original_messages) * 100
    
    def log(self):
        """Log formatted stats."""
        if self.original_messages == self.reduced_messages:
            logger.info(f"DYCP | No reduction needed ({self.original_messages} messages, ~{self.original_tokens:,} tokens)")
            return
        
        logger.info("=" * 70)
        logger.info("DYCP Context Reduction Summary")
        logger.info("=" * 70)
        logger.info(f"Messages:     {self.original_messages} -> {self.reduced_messages} ({self.message_reduction_percent:.1f}% reduction)")
        logger.info(f"Tokens:       ~{self.original_tokens:,} -> ~{self.reduced_tokens:,} ({self.reduction_percent:.1f}% reduction)")
        logger.info(f"Tokens Saved: ~{self.tokens_saved:,}")
        logger.info("-" * 70)
        logger.info(f"Spans Found:  {self.spans_found}")
        for i, (start, end) in enumerate(self.span_details):
            logger.info(f"  Span {i+1}: messages {start}-{end} ({end - start + 1} messages)")
        logger.info(f"Ghost Graph:  {self.ghost_graph_entities} entities, {self.ghost_graph_boosts} boosts applied")
        logger.info(f"Decay Lambda: {self.decay_lambda}")
        logger.info(f"Processing:   {self.processing_time_ms:.2f}ms")
        logger.info("=" * 70)


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens in message list."""
    total = 0
    for m in messages:
        # Role + content + message overhead (~4 tokens)
        total += estimate_tokens(m.get("content", "")) + 4
    return total

router = APIRouter()

# Global components (lazy loaded)
_components = {}


def _get_components(api_key: str = None, embedding_model: str = None, base_url: str = None):
    """Get or initialize components."""
    if "dycp" not in _components:
        _components["dycp"] = DYCPCore()
        _components["ghost_graphs"] = {}  # Per-conversation
        _components["milvus"] = get_milvus_storage()
    
    # Embedding provider (may need API key, model, and base_url)
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    url_suffix = f"_{hash(base_url) % 10000}" if base_url else ""
    key = "embeddings" if api_key is None else f"embeddings_{api_key[:8]}{model_suffix}{url_suffix}"
    if key not in _components:
        _components[key] = get_embedding_provider(api_key=api_key, model=embedding_model, base_url=base_url)
    
    return _components


def _get_conversation_id(messages: List[Dict]) -> int:
    """
    Generate a stable conversation ID from message history.
    Uses hash of first few messages to identify returning conversations.
    """
    if not messages:
        return 0
    
    # Use first 3 messages (or fewer) to create stable ID
    seed_messages = messages[:3]
    seed_str = json.dumps(seed_messages, sort_keys=True)
    hash_hex = hashlib.sha256(seed_str.encode()).hexdigest()[:12]
    
    # Convert to integer (for Milvus compatibility)
    return int(hash_hex, 16) % (10**9)


def _get_ghost_graph(conversation_id: int) -> GhostGraph:
    """Get or create Ghost Graph for a conversation."""
    components = _get_components()
    
    if conversation_id not in components["ghost_graphs"]:
        components["ghost_graphs"][conversation_id] = GhostGraph()
    
    return components["ghost_graphs"][conversation_id]


# Import OpenAI-compatible models
from .models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse
)

# Extended request model for proxy (adds enhanced_mode and conversation_id)
from pydantic import BaseModel
from typing import List as TypingList

class ProxyChatCompletionRequest(BaseModel):
    """Extended chat completion request with Lethus-specific fields"""
    model: str
    messages: TypingList[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[TypingList[str]] = None
    user: Optional[str] = None
    user_id: Optional[str] = None  # For database API key lookup
    conversation_id: Optional[int] = None  # For fetching enhanced_mode from DB
    enhanced_mode: Optional[bool] = None  # Override: True = DYCP, False = passthrough


# === Core DYCP Logic ===

def apply_dycp_reduction(
    messages: List[Dict],
    api_key: str,
    conversation_id: int,
    embedding_model: str = None,
    base_url: str = None
) -> tuple[List[Dict], DYCPStats]:
    """
    Apply DYCP to reduce message history to relevant spans only.
    
    Takes full conversation history from client, returns pruned version and stats.
    """
    start_time = time.time()
    stats = DYCPStats()
    stats.original_messages = len(messages)
    stats.original_tokens = estimate_messages_tokens(messages)
    stats.decay_lambda = settings.decay_lambda
    
    if len(messages) <= 2:
        # Too short to prune - just pass through
        stats.reduced_messages = len(messages)
        stats.reduced_tokens = stats.original_tokens
        stats.processing_time_ms = (time.time() - start_time) * 1000
        return messages, stats
    
    components = _get_components(api_key, embedding_model, base_url)
    dycp = components["dycp"]
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    url_suffix = f"_{hash(base_url) % 10000}" if base_url else ""
    embeddings = components.get(f"embeddings_{api_key[:8]}{model_suffix}{url_suffix}" if api_key else "embeddings")
    if not embeddings:
        embeddings = get_embedding_provider(api_key=api_key, model=embedding_model, base_url=base_url)
    
    ghost_graph = _get_ghost_graph(conversation_id)
    
    # Separate system prompt from conversation
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    if len(conversation) <= 2:
        stats.reduced_messages = len(messages)
        stats.reduced_tokens = stats.original_tokens
        stats.processing_time_ms = (time.time() - start_time) * 1000
        return messages, stats
    
    # Get the current query (last user message)
    last_user_msg = None
    for m in reversed(conversation):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    
    if not last_user_msg:
        stats.reduced_messages = len(messages)
        stats.reduced_tokens = stats.original_tokens
        stats.processing_time_ms = (time.time() - start_time) * 1000
        return messages, stats
    
    # Build embeddings for conversation history (excluding last message)
    history = conversation[:-1]
    if not history:
        stats.reduced_messages = len(messages)
        stats.reduced_tokens = stats.original_tokens
        stats.processing_time_ms = (time.time() - start_time) * 1000
        return messages, stats
    
    # Generate embeddings
    history_texts = [m["content"] for m in history]
    history_embs = np.array([embeddings.embed_text(t) for t in history_texts])
    query_emb = embeddings.embed_text(last_user_msg)
    
    # Create turn indices
    turn_indices = np.arange(len(history))
    current_turn = len(history)
    
    # Compute relevance with decay
    similarities = dycp.compute_relevance(
        query_emb,
        history_embs,
        turn_indices=turn_indices,
        current_turn=current_turn,
        apply_decay=True
    )
    
    # Ghost Graph boost
    history_with_entities = []
    total_entities = 0
    all_entity_names = []
    for i, m in enumerate(history):
        entities = ghost_graph.extract_entities(m["content"])
        total_entities += len(entities)
        entity_names = [e["name"] for e in entities]
        all_entity_names.extend(entity_names)
        history_with_entities.append({
            "role": m["role"],
            "content": m["content"],
            "turn_index": i,
            "entities": json.dumps(entity_names)
        })
    
    stats.ghost_graph_entities = total_entities
    stats.entity_names = list(set(all_entity_names))  # Unique entities
    
    # Track boosts
    original_similarities = similarities.copy()
    similarities = ghost_graph.boost_similarities(
        last_user_msg,
        history_with_entities,
        similarities
    )
    
    # Calculate boost details
    boost_details = []
    for i, (orig, boosted) in enumerate(zip(original_similarities, similarities)):
        if boosted > orig:
            boost_details.append({
                "turn": i,
                "original": float(orig),
                "boosted": float(boosted),
                "delta": float(boosted - orig)
            })
    
    stats.ghost_graph_boosts = len(boost_details)
    stats.boost_details = boost_details
    stats.similarity_scores = [float(s) for s in similarities]
    stats.tau = settings.dycp_tau
    stats.theta = settings.dycp_theta
    stats.enhanced_mode = True
    
    # Get spans using Kadane's Algorithm
    spans = dycp.get_pruned_indices(similarities)
    stats.spans_found = len(spans)
    stats.span_details = spans
    
    # Build reduced message list
    reduced_messages = list(system_messages)  # Keep system prompts
    
    if spans:
        # Add only messages from selected spans
        selected_indices = set()
        for start, end in spans:
            for i in range(start, end + 1):
                selected_indices.add(i)
        
        for i, m in enumerate(history):
            if i in selected_indices:
                reduced_messages.append(m)
    
    # Always include the last message (current query)
    reduced_messages.append(conversation[-1])
    
    # Final stats
    stats.reduced_messages = len(reduced_messages)
    stats.reduced_tokens = estimate_messages_tokens(reduced_messages)
    stats.processing_time_ms = (time.time() - start_time) * 1000
    
    return reduced_messages, stats


async def store_interaction(
    conversation_id: int,
    user_message: str,
    assistant_message: str,
    api_key: str,
    embedding_model: str = None,
    base_url: str = None
):
    # Store the conversation turn for future retrieval.
    components = _get_components(api_key, embedding_model, base_url)
    milvus = components["milvus"]
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    url_suffix = f"_{hash(base_url) % 10000}" if base_url else ""
    embeddings = components.get(f"embeddings_{api_key[:8]}{model_suffix}{url_suffix}" if api_key else "embeddings")
    if not embeddings:
        embeddings = get_embedding_provider(api_key=api_key, model=embedding_model, base_url=base_url)
    
    ghost_graph = _get_ghost_graph(conversation_id)
    
    # Extract and register entities
    user_entities = ghost_graph.extract_entities(user_message)
    assistant_entities = ghost_graph.extract_entities(assistant_message)
    all_entity_names = [e["name"] for e in user_entities + assistant_entities]
    
    ghost_graph.register_entities(user_entities + assistant_entities)
    
    # Store combined turn in Milvus
    combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
    embedding = embeddings.embed_text(combined_text)
    
    milvus.add_turn(
        role="combined",
        content=combined_text,
        embedding=embedding,
        conversation_id=conversation_id,
        entities=json.dumps(all_entity_names)
    )


async def stream_and_capture(
    target_url: str,
    forward_body: dict,
    headers: dict,
    conversation_id: int,
    user_message: str,
    api_key: str,
    embedding_model: str = None,
    base_url: str = None
) -> AsyncIterator[bytes]:
    """
    Stream response to client while capturing full content for storage.
    Creates its own httpx client to manage lifecycle properly.
    """
    full_content = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            target_url,
            json=forward_body,
            headers=headers
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"data: {json.dumps({'error': error_body.decode()})}\n\n".encode()
                return
            
            async for chunk in response.aiter_bytes():
                yield chunk
                
                # Parse SSE chunks to capture content
                try:
                    chunk_str = chunk.decode('utf-8')
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_content.append(delta['content'])
                except:
                    pass
    
    # Store the complete interaction after stream closes
    if full_content and user_message:
        assistant_message = ''.join(full_content)
        await store_interaction(
            conversation_id,
            user_message,
            assistant_message,
            api_key,
            embedding_model,
            base_url
        )


# === API Endpoints ===

@router.post("/chat/completions")
async def chat_completions(request: Request, db: Session = Depends(get_db)):
    """
    OpenAI-compatible chat completions endpoint.
    Applies DYCP context reduction before forwarding to LLM.
    
    Accepts API key via:
    1. Authorization header (Bearer token) - for standard OpenAI clients
    2. user_id in request body - fetches API key from database
    
    Settings priority (user API settings > env defaults):
    - Model, temperature, max_tokens, base_url from user settings
    - Falls back to env defaults if not set
    """
    # Parse request body first
    body = await request.json()
    
    # Get API key and user settings from header or database
    auth_header = request.headers.get("Authorization", "")
    api_key = None
    user_llm_model = None
    user_llm_temperature = None
    user_llm_max_tokens = None
    user_embedding_model = None
    user_embedding_dim = None
    user_base_url = None
    
    if auth_header.startswith("Bearer "):
        # Standard OpenAI client with API key in header
        api_key = auth_header[7:]  # Remove "Bearer "
        user_base_url = None
    elif "user_id" in body:
        # Fetch API key and settings from database using user_id
        user_id = body.get("user_id")
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user or not user.openai_api_key:
            raise HTTPException(
                status_code=400,
                detail="No API key configured. Please set your OpenAI API key in settings."
            )
        
        api_key = user.openai_api_key
        user_base_url = user.openai_base_url
        user_llm_model = user.llm_model
        user_llm_temperature = user.llm_temperature
        user_llm_max_tokens = user.llm_max_tokens
        user_embedding_model = user.embedding_model
        user_embedding_dim = user.embedding_dim
    else:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization header or user_id in request body."
        )
    
    # Parse and validate request
    try:
        chat_request = ProxyChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    
    # Determine enhanced_mode from request or database
    # Priority: request body > conversation in DB > default (True)
    enhanced_mode = True  # Default
    db_conversation_id = chat_request.conversation_id
    
    if chat_request.enhanced_mode is not None:
        # Explicit override in request
        enhanced_mode = chat_request.enhanced_mode
    elif db_conversation_id:
        # Look up from database
        conv = db.query(Conversation).filter(Conversation.id == db_conversation_id).first()
        if conv:
            enhanced_mode = conv.enhanced_mode if conv.enhanced_mode is not None else True
    
    # Determine effective settings (user API settings > request > env defaults)
    # Model: request model > user setting > env default
    effective_llm_model = chat_request.model
    if not effective_llm_model or effective_llm_model == "default":
        effective_llm_model = user_llm_model or settings.llm_model
    
    # Temperature: user setting > request > env default
    effective_temperature = user_llm_temperature if user_llm_temperature is not None else (
        chat_request.temperature if chat_request.temperature is not None else settings.llm_temperature
    )
    
    # Max tokens: user setting > request > env default
    effective_max_tokens = user_llm_max_tokens if user_llm_max_tokens is not None else (
        chat_request.max_tokens if chat_request.max_tokens is not None else settings.llm_max_tokens
    )
    
    # Embedding model: user setting > env default
    effective_embedding_model = user_embedding_model or settings.openai_embedding_model
    
    # Embedding dim: user setting > env default
    effective_embedding_dim = user_embedding_dim or settings.openai_embedding_dim
    
    # Base URL: user setting > env default
    effective_base_url = user_base_url or settings.openai_base_url
    
    # Convert to dicts for processing
    messages = [{"role": m.role, "content": m.content} for m in chat_request.messages]
    
    # Get conversation ID (use DB conversation_id if provided, else generate from messages)
    conversation_id = db_conversation_id or _get_conversation_id(messages)
    
    # Get last user message for storage later
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    
    # Apply DYCP reduction only if enhanced_mode is True
    if enhanced_mode:
        reduced_messages, stats = apply_dycp_reduction(
            messages, api_key, conversation_id, effective_embedding_model, effective_base_url
        )
        stats.enhanced_mode = True
        # Log detailed stats
        stats.log()
    else:
        # Passthrough mode - no DYCP, just forward messages as-is
        stats = DYCPStats()
        stats.original_messages = len(messages)
        stats.reduced_messages = len(messages)
        stats.original_tokens = estimate_messages_tokens(messages)
        stats.reduced_tokens = stats.original_tokens
        stats.enhanced_mode = False
        reduced_messages = messages
        logger.info("DYCP | Passthrough mode (enhanced_mode=False)")
    
    # Build forwarded request - use effective settings
    forward_body = body.copy()
    forward_body["messages"] = reduced_messages
    forward_body["model"] = effective_llm_model
    forward_body["temperature"] = effective_temperature
    forward_body["max_tokens"] = effective_max_tokens
    forward_body.pop("user_id", None)
    forward_body.pop("conversation_id", None)
    forward_body.pop("enhanced_mode", None)
    
    # Determine target URL based on model
    model = effective_llm_model.lower()
    if "claude" in model or "anthropic" in model:
        # Anthropic - would need different handling
        raise HTTPException(
            status_code=400, 
            detail="Anthropic models not yet supported. Use OpenAI models."
        )
    else:
        # OpenAI (default) - use the effective_base_url determined earlier
        target_url = f"{effective_base_url.rstrip('/')}/chat/completions"
    
    # Forward request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # DYCP stats headers (basic stats always included)
    dycp_headers = {
        "X-Lethus-Enhanced-Mode": str(stats.enhanced_mode).lower(),
        "X-Lethus-Original-Messages": str(stats.original_messages),
        "X-Lethus-Reduced-Messages": str(stats.reduced_messages),
        "X-Lethus-Original-Tokens": str(stats.original_tokens),
        "X-Lethus-Reduced-Tokens": str(stats.reduced_tokens),
        "X-Lethus-Tokens-Saved": str(stats.tokens_saved),
        "X-Lethus-Reduction-Percent": f"{stats.reduction_percent:.1f}",
        "X-Lethus-Spans-Found": str(stats.spans_found),
        "X-Lethus-Processing-Ms": f"{stats.processing_time_ms:.2f}",
    }
    
    # Extended stats for enhanced mode (dev logs)
    if stats.enhanced_mode:
        dycp_headers.update({
            "X-Lethus-Ghost-Entities": str(stats.ghost_graph_entities),
            "X-Lethus-Ghost-Boosts": str(stats.ghost_graph_boosts),
            "X-Lethus-Decay-Lambda": f"{stats.decay_lambda:.3f}",
            "X-Lethus-Tau": f"{stats.tau:.3f}",
            "X-Lethus-Theta": f"{stats.theta:.3f}",
            # JSON-encoded extended data (for detailed dropdown)
            "X-Lethus-Entity-Names": json.dumps(stats.entity_names[:20]),  # Limit to 20 entities
            "X-Lethus-Span-Details": json.dumps(stats.span_details),
            "X-Lethus-Boost-Count": str(len(stats.boost_details)),
        })
    
    if chat_request.stream:
        # Streaming response - generator creates its own client
        return StreamingResponse(
            stream_and_capture(
                target_url,
                forward_body,
                headers,
                conversation_id,
                last_user_msg,
                api_key,
                effective_embedding_model,
                effective_base_url
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable NGINX buffering
                "Transfer-Encoding": "chunked",
                **dycp_headers
            }
        )
    
    # Non-streaming response
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            target_url,
            json=forward_body,
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        result = response.json()
        
        # Store interaction
        if last_user_msg and result.get("choices"):
            assistant_msg = result["choices"][0]["message"]["content"]
            await store_interaction(
                conversation_id,
                last_user_msg,
                assistant_msg,
                api_key,
                effective_embedding_model,
                effective_base_url
            )
        
        # Return with DYCP stats headers
        return JSONResponse(content=result, headers=dycp_headers)


@router.get("/models")
async def list_models(request: Request):
    """List available models (passthrough to OpenAI)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = auth_header[7:]
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.openai_base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request):
    """Get model details (passthrough to OpenAI)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = auth_header[7:]
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.openai_base_url}/models/{model_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()
