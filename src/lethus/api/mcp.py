"""
Lethus MCP Server.
Exposes DYCP memory as MCP tools for LLM integration.
"""
from mcp.server.fastmcp import FastMCP
import numpy as np
import json
import os

from ..config import settings
from ..core.dycp import DYCPCore
from ..core.ghost_graph import GhostGraph
from ..core.prefetch import PrefetchCache, Prefetcher
from ..core.embeddings import get_embedding_provider
from ..storage.milvus import get_milvus_storage

# Initialize MCP server
mcp = FastMCP("lethus")

# Initialize components (lazy loaded on first use)
_components = {}


def _get_components():
    """Lazy initialize all components"""
    if not _components:
        print("Initializing Lethus MCP Server...")
        print(f"  Milvus URI: {settings.milvus_uri}")
        print(f"  Decay Lambda: {settings.decay_lambda}")
        print(f"  Embedding Provider: {settings.embedding_provider}")
        
        _components["storage"] = get_milvus_storage()
        _components["embeddings"] = get_embedding_provider()
        _components["dycp"] = DYCPCore()
        _components["ghost_graph"] = GhostGraph()
        _components["prefetch_cache"] = PrefetchCache()
        
        # Setup prefetcher with compute function
        def compute_context(query: str):
            return _compute_context_internal(query)
        
        _components["prefetcher"] = Prefetcher(
            cache=_components["prefetch_cache"],
            compute_fn=compute_context
        )
        
        print("Lethus MCP Server Ready.")
    
    return _components


def _compute_context_internal(query: str):
    """Internal function to compute DYCP context for a query."""
    c = _get_components()
    
    turns, history_embs = c["storage"].get_all_turns_ordered()
    
    if not turns:
        return [], "No memory available."
    
    query_emb = c["embeddings"].embed_text(query)
    current_turn = len(turns)
    turn_indices = np.array([t["turn_index"] for t in turns])
    
    # Compute relevance with semantic decay
    similarities = c["dycp"].compute_relevance(
        query_emb,
        history_embs,
        turn_indices=turn_indices,
        current_turn=current_turn,
        apply_decay=True
    )
    
    # Ghost Graph augmentation
    similarities = c["ghost_graph"].boost_similarities(query, turns, similarities)
    
    # Get spans using Kadane's Algorithm
    spans = c["dycp"].get_pruned_indices(similarities)
    
    if not spans:
        return [], "No relevant context found in memory."
    
    # Format context
    context_str = c["dycp"].format_context(
        turns, spans, 
        header="RELEVANT CONTEXT (DYCP + Decay + Ghost Graph)"
    )
    
    return spans, context_str


@mcp.tool()
def store_interaction(role: str, content: str) -> str:
    """
    Stores a conversation turn in long-term memory with entity extraction.
    Call this after EVERY user message and assistant response.
    
    Args:
        role: Either "user" or "assistant"
        content: The full text content of the message
    """
    c = _get_components()
    
    # Generate embedding
    embedding = c["embeddings"].embed_text(content)
    
    # Extract entities for Ghost Graph
    entities = c["ghost_graph"].extract_entities(content)
    entity_names = [e["name"] for e in entities]
    
    # Register in Ghost Graph
    c["ghost_graph"].register_entities(entities)
    
    # Store in Milvus
    c["storage"].add_turn(role, content, embedding, entities=json.dumps(entity_names))
    
    entity_summary = f" Entities: {entity_names}" if entity_names else ""
    return f"Stored in memory.{entity_summary}"


@mcp.tool()
def get_context(query: str) -> str:
    """
    Retrieves relevant conversation context using DYCP with Semantic Decay.
    
    Uses:
    1. Dynamic span selection (Kadane's Algorithm) - finds coherent conversation blocks
    2. Semantic Decay - older messages need higher relevance to be recalled
    3. Ghost Graph - entity linking boosts related context
    
    Args:
        query: The user's current question or topic
    """
    c = _get_components()
    
    # Check prefetch cache first
    cached = c["prefetcher"].get_cached_or_none(query)
    if cached:
        return cached
    
    _, context_str = _compute_context_internal(query)
    return context_str


@mcp.tool()
def search_by_entity(entity_name: str) -> str:
    """
    Search memory for turns mentioning a specific entity.
    Useful when you know the exact name of a person, config, or concept.
    
    Args:
        entity_name: The entity to search for (e.g., "API_KEY", "John", "AWS")
    """
    c = _get_components()
    
    results = c["storage"].search_by_entity(entity_name, limit=10)
    
    if not results:
        return f"No memory found mentioning '{entity_name}'."
    
    context_str = f"--- MEMORY SEARCH: '{entity_name}' ---\n"
    for r in results:
        context_str += f"[{r['role'].upper()}]: {r['content']}\n\n"
    context_str += "--- END SEARCH ---"
    
    return context_str


@mcp.tool()
def get_entity_graph() -> str:
    """
    Returns the current state of the Ghost Graph (entity relationships).
    Useful for debugging or understanding what entities are tracked.
    """
    c = _get_components()
    return c["ghost_graph"].get_state_summary()


@mcp.tool()
def clear_memory() -> str:
    """Wipes all conversation history and resets the Ghost Graph."""
    c = _get_components()
    
    c["storage"].clear_all()
    c["ghost_graph"].clear()
    c["prefetch_cache"].clear()
    
    return "All memory cleared."


@mcp.tool()
def get_memory_stats() -> str:
    """Returns statistics about the current memory state."""
    c = _get_components()
    
    turn_count = c["storage"].get_turn_count()
    entity_count = len(c["ghost_graph"].entities)
    cache_size = len(c["prefetch_cache"])
    
    return (
        f"Memory Stats:\n"
        f"  Total Turns: {turn_count}\n"
        f"  Tracked Entities: {entity_count}\n"
        f"  Prefetch Cache Size: {cache_size}\n"
        f"  Decay Lambda: {settings.decay_lambda}\n"
        f"  DYCP Tau: {settings.dycp_tau}\n"
        f"  DYCP Theta: {settings.dycp_theta}"
    )


def main():
    """Entry point for MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
