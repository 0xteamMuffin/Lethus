"""
Predictive Prefetching: Pre-compute context spans for likely follow-up queries.
Hides retrieval latency by warming the cache while user reads the response.
"""
import asyncio
from typing import List, Callable, Optional, Any
from dataclasses import dataclass
import time
from collections import OrderedDict

from ..config import settings


@dataclass
class CachedSpan:
    """Cached context span result"""
    query: str
    spans: List[tuple]  # List of (start, end) indices
    context_str: str  # Pre-formatted context
    created_at: float
    ttl: float = None
    
    def __post_init__(self):
        if self.ttl is None:
            self.ttl = settings.prefetch_ttl
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class PrefetchCache:
    """
    LRU cache for pre-computed context spans.
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.prefetch_cache_size
        self._cache: OrderedDict[str, CachedSpan] = OrderedDict()
    
    def _make_key(self, query: str) -> str:
        """Simple hash for cache key."""
        return str(hash(query.lower().strip()))
    
    def get(self, query: str) -> Optional[CachedSpan]:
        """Get cached result for query, or None if not found/expired."""
        key = self._make_key(query)
        if key in self._cache:
            span = self._cache[key]
            if not span.is_expired():
                self._cache.move_to_end(key)  # LRU update
                return span
            else:
                del self._cache[key]
        return None
    
    def put(self, query: str, spans: List[tuple], context_str: str):
        """Store result in cache."""
        key = self._make_key(query)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = CachedSpan(
            query=query,
            spans=spans,
            context_str=context_str,
            created_at=time.time()
        )
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class Prefetcher:
    """
    Async prefetcher that predicts follow-up queries and warms the cache.
    """
    
    def __init__(
        self,
        cache: PrefetchCache,
        compute_fn: Callable[[str], tuple],
        predict_fn: Optional[Callable[[str, str], List[str]]] = None
    ):
        """
        Args:
            cache: The span cache
            compute_fn: Function to compute spans: fn(query) -> (spans, context_str)
            predict_fn: Optional function to predict follow-ups: fn(query, response) -> [queries]
        """
        self.cache = cache
        self.compute_fn = compute_fn
        self.predict_fn = predict_fn or self._default_predictions
        self._background_tasks: List[asyncio.Task] = []
    
    def _default_predictions(self, last_query: str, last_response: str) -> List[str]:
        """
        Smarter follow-up predictions based on query structure and entities.
        """
        predictions = []
        
        # Extract key terms from the query and response
        query_lower = last_query.lower()
        
        # Pattern 1: If query was about "how", predict "why" and "what if"
        if query_lower.startswith(("how do", "how to", "how can")):
            predictions.append(last_query.replace("how", "why", 1))
            predictions.append("What if that doesn't work?")
        
        # Pattern 2: If query was about "what is", predict usage questions
        if query_lower.startswith(("what is", "what's", "what are")):
            topic = last_query.split(maxsplit=2)[-1].rstrip("?")
            predictions.append(f"How do I use {topic}?")
            predictions.append(f"Show me an example of {topic}")
        
        # Pattern 3: Extract entities from response for entity-based predictions
        try:
            from .ghost_graph import GhostGraph
            ghost = GhostGraph(use_spacy=True)
            entities = ghost.extract_entities(last_response)
            
            # Filter to meaningful entities (PERSON, ORG, PRODUCT, CONFIG, etc.)
            important_types = {"PERSON", "ORG", "PRODUCT", "CONFIG", "FUNCTION", "GPE"}
            important_entities = [e for e in entities if e["type"] in important_types]
            
            for ent in important_entities[:3]:
                name = ent["name"]
                if name.lower() not in query_lower:  # Don't repeat what was asked
                    predictions.append(f"Tell me more about {name}")
                    predictions.append(f"What is {name}?")
        except Exception:
            # Fall back to simple capitalized word extraction
            response_words = last_response.split()
            capitalized = [w.strip(".,!?") for w in response_words 
                          if w and w[0].isupper() and len(w) > 3 and w.lower() not in query_lower]
            for word in capitalized[:2]:
                predictions.append(f"What is {word}?")
        
        # Pattern 4: Context continuation patterns
        if "database" in query_lower or "db" in query_lower:
            predictions.extend(["What's the database connection string?", "How do I connect to the database?"])
        if "api" in query_lower:
            predictions.extend(["What endpoints are available?", "How do I authenticate?"])
        if "error" in query_lower or "bug" in query_lower:
            predictions.extend(["How do I fix it?", "What's causing this?"])
        if "deploy" in query_lower:
            predictions.extend(["What are the deployment steps?", "What environment variables do I need?"])
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for p in predictions:
            p_lower = p.lower()
            if p_lower not in seen:
                seen.add(p_lower)
                unique.append(p)
        
        return unique[:5]
    
    async def trigger_prefetch(self, last_query: str, last_response: str):
        """
        Called after generating a response. Starts background prefetch.
        """
        if not settings.prefetch_enabled:
            return
        
        predicted = self.predict_fn(last_query, last_response)
        
        for query in predicted:
            if self.cache.get(query) is None:
                task = asyncio.create_task(self._prefetch_one(query))
                self._background_tasks.append(task)
        
        # Cleanup completed tasks
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
    
    async def _prefetch_one(self, query: str):
        """Compute and cache spans for one predicted query."""
        try:
            loop = asyncio.get_event_loop()
            spans, context_str = await loop.run_in_executor(None, self.compute_fn, query)
            self.cache.put(query, spans, context_str)
        except Exception:
            pass  # Prefetch is best-effort
    
    def get_cached_or_none(self, query: str) -> Optional[str]:
        """
        Check if we have a cached result for this query.
        Returns the pre-formatted context string or None.
        """
        cached = self.cache.get(query)
        return cached.context_str if cached else None
