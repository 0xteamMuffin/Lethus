"""
Lethus OpenAI-Compatible Proxy.
Drop-in replacement for OpenAI API that applies DYCP context reduction.

Usage:
    Set base_url="http://localhost:8000/v1" in any OpenAI-compatible client.
    Your API key is passed through to the real LLM provider.
"""
import json
import hashlib
from typing import List, Optional, Dict, AsyncIterator
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import numpy as np

from ..config import settings
from ..storage.milvus import get_milvus_storage
from ..core.dycp import DYCPCore
from ..core.ghost_graph import GhostGraph
from ..core.embeddings import get_embedding_provider

router = APIRouter()

# Global components (lazy loaded)
_components = {}


def _get_components(api_key: str = None, embedding_model: str = None):
    """Get or initialize components."""
    if "dycp" not in _components:
        _components["dycp"] = DYCPCore()
        _components["ghost_graphs"] = {}  # Per-conversation
        _components["milvus"] = get_milvus_storage()
    
    # Embedding provider (may need API key and model)
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    key = "embeddings" if api_key is None else f"embeddings_{api_key[:8]}{model_suffix}"
    if key not in _components:
        _components[key] = get_embedding_provider(api_key=api_key, model=embedding_model)
    
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


# === Core DYCP Logic ===

def apply_dycp_reduction(
    messages: List[Dict],
    api_key: str,
<<<<<<< HEAD
    conversation_id: int
) -> List[Dict]:
=======
    conversation_id: int,
    embedding_model: str = None
) -> tuple[List[Dict], DYCPStats]:
>>>>>>> main
    """
    Apply DYCP to reduce message history to relevant spans only.
    
    Takes full conversation history from client, returns pruned version.
    """
    if len(messages) <= 2:
        # Too short to prune - just pass through
        return messages
    
    components = _get_components(api_key, embedding_model)
    dycp = components["dycp"]
<<<<<<< HEAD
    milvus = components["milvus"]
    embeddings = components.get(f"embeddings_{api_key[:8]}" if api_key else "embeddings")
=======
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    embeddings = components.get(f"embeddings_{api_key[:8]}{model_suffix}" if api_key else "embeddings")
>>>>>>> main
    if not embeddings:
        embeddings = get_embedding_provider(api_key=api_key, model=embedding_model)
    
    ghost_graph = _get_ghost_graph(conversation_id)
    
    # Separate system prompt from conversation
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    
    if len(conversation) <= 2:
        return messages
    
    # Get the current query (last user message)
    last_user_msg = None
    for m in reversed(conversation):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    
    if not last_user_msg:
        return messages
    
    # Build embeddings for conversation history (excluding last message)
    history = conversation[:-1]
    if not history:
        return messages
    
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
    for i, m in enumerate(history):
        entities = ghost_graph.extract_entities(m["content"])
        history_with_entities.append({
            "role": m["role"],
            "content": m["content"],
            "turn_index": i,
            "entities": json.dumps([e["name"] for e in entities])
        })
    
    similarities = ghost_graph.boost_similarities(
        last_user_msg,
        history_with_entities,
        similarities
    )
    
    # Get spans using Kadane's Algorithm
    spans = dycp.get_pruned_indices(similarities)
    
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
    
    return reduced_messages


async def store_interaction(
    conversation_id: int,
    user_message: str,
    assistant_message: str,
    api_key: str,
    embedding_model: str = None
):
    """Store the conversation turn for future retrieval."""
    components = _get_components(api_key, embedding_model)
    milvus = components["milvus"]
    model_suffix = f"_{embedding_model}" if embedding_model else ""
    embeddings = components.get(f"embeddings_{api_key[:8]}{model_suffix}" if api_key else "embeddings")
    if not embeddings:
        embeddings = get_embedding_provider(api_key=api_key, model=embedding_model)
    
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
    response: httpx.Response,
    conversation_id: int,
    user_message: str,
    api_key: str,
    embedding_model: str = None
) -> AsyncIterator[bytes]:
    """
    Stream response to client while capturing full content for storage.
    """
    full_content = []
    
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
    
    # Store the complete interaction
    if full_content:
        assistant_message = ''.join(full_content)
        await store_interaction(
            conversation_id,
            user_message,
            assistant_message,
            api_key,
            embedding_model
        )


# === API Endpoints ===

@router.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Applies DYCP context reduction before forwarding to LLM.
<<<<<<< HEAD
=======
    
    Accepts API key via:
    1. Authorization header (Bearer token) - for standard OpenAI clients
    2. user_id in request body - fetches API key from database
    
    Model selection:
    - Uses model from request if specified
    - Falls back to user's saved llm_model preference
    - Falls back to env default (settings.llm_model)
>>>>>>> main
    """
    # Get API key from header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = auth_header[7:]  # Remove "Bearer "
    
    # Parse request body
    body = await request.json()
    
<<<<<<< HEAD
=======
    # Get API key and user settings from header or database
    auth_header = request.headers.get("Authorization", "")
    api_key = None
    user_llm_model = None
    user_embedding_model = None
    
    if auth_header.startswith("Bearer "):
        # Standard OpenAI client with API key in header
        api_key = auth_header[7:]  # Remove "Bearer "
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
        user_llm_model = user.llm_model
        user_embedding_model = user.embedding_model
    else:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization header or user_id in request body."
        )
    
    # Parse and validate request
>>>>>>> main
    try:
        chat_request = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    
    # Determine which LLM model to use (request model > user setting > env default)
    effective_llm_model = chat_request.model
    if not effective_llm_model or effective_llm_model == "default":
        effective_llm_model = user_llm_model or settings.llm_model
    
    # Determine which embedding model to use (user setting > env default)
    effective_embedding_model = user_embedding_model or settings.openai_embedding_model
    
    # Convert to dicts for processing
    messages = [{"role": m.role, "content": m.content} for m in chat_request.messages]
    
    # Get conversation ID
    conversation_id = _get_conversation_id(messages)
    
    # Get last user message for storage later
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    
<<<<<<< HEAD
    # Apply DYCP reduction
    reduced_messages = apply_dycp_reduction(messages, api_key, conversation_id)
=======
    # Apply DYCP reduction with user's embedding model
    reduced_messages, stats = apply_dycp_reduction(
        messages, api_key, conversation_id, effective_embedding_model
    )
>>>>>>> main
    
    # Log reduction stats
    original_count = len(messages)
    reduced_count = len(reduced_messages)
    if original_count != reduced_count:
        print(f"[DYCP] Reduced {original_count} → {reduced_count} messages")
    
<<<<<<< HEAD
    # Build forwarded request
    forward_body = body.copy()
    forward_body["messages"] = reduced_messages
=======
    # Build forwarded request - remove user_id, use effective model
    forward_body = body.copy()
    forward_body["messages"] = reduced_messages
    forward_body["model"] = effective_llm_model
    forward_body.pop("user_id", None)
>>>>>>> main
    
    # Determine target URL based on model
    model = effective_llm_model.lower()
    if "claude" in model or "anthropic" in model:
        # Anthropic - would need different handling
        raise HTTPException(
            status_code=400, 
            detail="Anthropic models not yet supported. Use OpenAI models."
        )
    else:
        # OpenAI (default)
        target_url = f"{settings.openai_base_url}/chat/completions"
    
    # Forward request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
<<<<<<< HEAD
=======
    # DYCP stats headers
    dycp_headers = {
        "X-Lethus-Original-Messages": str(stats.original_messages),
        "X-Lethus-Reduced-Messages": str(stats.reduced_messages),
        "X-Lethus-Original-Tokens": str(stats.original_tokens),
        "X-Lethus-Reduced-Tokens": str(stats.reduced_tokens),
        "X-Lethus-Tokens-Saved": str(stats.tokens_saved),
        "X-Lethus-Reduction-Percent": f"{stats.reduction_percent:.1f}",
        "X-Lethus-Spans-Found": str(stats.spans_found),
        "X-Lethus-Processing-Ms": f"{stats.processing_time_ms:.2f}",
    }
    
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
                effective_embedding_model
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                **dycp_headers
            }
        )
    
    # Non-streaming response
>>>>>>> main
    async with httpx.AsyncClient(timeout=120.0) as client:
        if chat_request.stream:
            # Streaming response
            async with client.stream(
                "POST",
                target_url,
                json=forward_body,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=error_body.decode()
                    )
                
                return StreamingResponse(
                    stream_and_capture(
                        response,
                        conversation_id,
                        last_user_msg,
                        api_key
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
        else:
            # Non-streaming response
            response = await client.post(
                target_url,
                json=forward_body,
                headers=headers
            )
<<<<<<< HEAD
            
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
                    api_key
                )
            
            return result
=======
        
        result = response.json()
        
        # Store interaction
        if last_user_msg and result.get("choices"):
            assistant_msg = result["choices"][0]["message"]["content"]
            await store_interaction(
                conversation_id,
                last_user_msg,
                assistant_msg,
                api_key,
                effective_embedding_model
            )
        
        # Return with DYCP stats headers
        return JSONResponse(content=result, headers=dycp_headers)
>>>>>>> main


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
