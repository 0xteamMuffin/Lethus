"""
Lethus - MCP Server with Long-Term Memory

Features:
- Dynamic span selection (Kadane's Algorithm)
- Semantic Decay (Time-weighted relevance)
- Ghost Graph (Entity linking for pronoun resolution)
- Predictive Prefetching (Cache warming)
"""
from mcp.server.fastmcp import FastMCP
from dycp_memory_server.dycp_logic import DYCPCore
from dycp_memory_server.storage import MilvusStorage
from dycp_memory_server.ghost_graph import GhostGraph
from dycp_memory_server.prefetch import PrefetchCache, Prefetcher
import numpy as np
import json
import os

# Configuration via environment variables
MILVUS_URI = os.environ.get("DYCP_MILVUS_URI", "http://localhost:19530")
DECAY_LAMBDA = float(os.environ.get("DYCP_DECAY_LAMBDA", "0.98"))
EMBEDDING_MODEL = os.environ.get("DYCP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Initialize components
mcp = FastMCP("lethus")

print(f"Initializing Lethus...")
print(f"  Milvus URI: {MILVUS_URI}")
print(f"  Decay Lambda: {DECAY_LAMBDA}")
print(f"  Embedding Model: {EMBEDDING_MODEL}")

storage = MilvusStorage(uri=MILVUS_URI)
core = DYCPCore(model_name=EMBEDDING_MODEL, decay_lambda=DECAY_LAMBDA)
ghost_graph = GhostGraph(use_spacy=True)
prefetch_cache = PrefetchCache(max_size=50)

print("Lethus Ready.")

def _compute_context(query: str) -> tuple:
    """
    Internal function to compute DYCP context for a query.
    Returns (spans, formatted_context_string).
    """
    turns, history_embs = storage.get_all_turns_ordered()
    
    if not turns:
        return [], "No memory available."
    
    query_emb = core.embed_text(query)
    current_turn = storage.get_current_turn_count()
    turn_indices = np.array([t["turn_index"] for t in turns])
    
    # Compute relevance with semantic decay
    similarities = core.compute_relevance(
        query_emb,
        history_embs,
        turn_indices=turn_indices,
        current_turn=current_turn,
        apply_decay=True
    )
    
    # Ghost Graph augmentation: boost scores for entity-linked turns
    mentioned_entities = ghost_graph.find_entity_mentions(query)
    if mentioned_entities:
        for entity in mentioned_entities:
            linked = ghost_graph.get_linked_entities(entity, depth=1)
            for i, turn in enumerate(turns):
                turn_entities = json.loads(turn.get("entities", "[]"))
                turn_entity_names = [e["name"] if isinstance(e, dict) else e for e in turn_entities]
                # Boost if this turn contains a linked entity
                for linked_ent in linked:
                    if linked_ent in turn_entity_names:
                        similarities[i] *= 1.2  # 20% boost
    
    # Get spans using Kadane's Algorithm
    spans = core.get_pruned_indices(similarities)
    
    if not spans:
        return [], "No relevant context found in memory."
    
    # Format context string
    context_str = "--- RELEVANT CONTEXT ---\n"
    
    prev_end = -1
    for start, end in spans:
        if start > prev_end + 1:
            context_str += "\n[...earlier context omitted...]\n\n"
        
        for i in range(start, end + 1):
            if i < len(turns):
                t = turns[i]
                context_str += f"[{t['role'].upper()}]: {t['content']}\n"
        
        prev_end = end
    
    context_str += "\n--- END CONTEXT ---"
    return spans, context_str

# Initialize prefetcher with compute function
prefetcher = Prefetcher(
    cache=prefetch_cache,
    compute_fn=_compute_context
)

@mcp.tool()
def store_interaction(role: str, content: str) -> str:
    """
    Stores a conversation turn in long-term memory with entity extraction.
    Call this after EVERY user message and assistant response.
    
    Args:
        role: Either "user" or "assistant"
        content: The full text content of the message
    """
    # Generate embedding
    embedding = core.embed_text(content)
    
    # Extract entities for Ghost Graph
    entities = ghost_graph.extract_entities(content)
    entity_names = [e["name"] for e in entities]
    
    # Register in Ghost Graph
    ghost_graph.register_entities(entities)
    
    # Store in Milvus
    storage.add_turn(role, content, embedding, entities=json.dumps(entity_names))
    
    entity_summary = f" Entities: {entity_names}" if entity_names else ""
    return f"Stored in memory.{entity_summary}"

@mcp.tool()
def get_context(query: str) -> str:
    """
    Retrieves relevant conversation context from long-term memory.
    
    Uses:
    1. Dynamic span selection - finds coherent conversation blocks
    2. Semantic Decay - older messages need higher relevance to be recalled
    3. Ghost Graph - entity linking boosts related context
    
    Args:
        query: The user's current question or topic
    """
    # Check prefetch cache first
    cached = prefetcher.get_cached_or_none(query)
    if cached:
        return cached
    
    _, context_str = _compute_context(query)
    return context_str

@mcp.tool()
def search_by_entity(entity_name: str) -> str:
    """
    Search memory for turns mentioning a specific entity.
    Useful when you know the exact name of a person, config, or concept.
    
    Args:
        entity_name: The entity to search for (e.g., "API_KEY", "John", "AWS")
    """
    results = storage.search_by_entity(entity_name, limit=10)
    
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
    if not ghost_graph.entities:
        return "Ghost Graph is empty. No entities tracked yet."
    
    lines = ["--- GHOST GRAPH STATE ---"]
    for name, entity in list(ghost_graph.entities.items())[:20]:  # Limit output
        links = list(entity.linked_entities)[:5]
        lines.append(f"  {name} ({entity.entity_type}) -> {links}")
    lines.append("--- END GRAPH ---")
    
    return "\n".join(lines)

@mcp.tool()
def clear_memory() -> str:
    """Wipes all conversation history and resets the Ghost Graph."""
    storage.clear_history()
    ghost_graph.entities.clear()
    prefetch_cache.clear()
    return "All memory cleared."

@mcp.tool()
def get_memory_stats() -> str:
    """Returns statistics about the current memory state."""
    turn_count = storage.get_current_turn_count()
    entity_count = len(ghost_graph.entities)
    cache_size = len(prefetch_cache._cache)
    
    return (
        f"Memory Stats:\n"
        f"  Total Turns: {turn_count}\n"
        f"  Tracked Entities: {entity_count}\n"
        f"  Prefetch Cache Size: {cache_size}\n"
        f"  Decay Lambda: {DECAY_LAMBDA}"
    )

def main():
    mcp.run()

if __name__ == "__main__":
    main()
