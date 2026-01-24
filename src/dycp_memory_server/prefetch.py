"""
Predictive Prefetching: Pre-compute context spans for likely follow-up queries.
Hides retrieval latency by warming the cache while user reads the response.
"""
import asyncio
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import time
from collections import OrderedDict

@dataclass
class CachedSpan:
    query: str
    spans: List[tuple]  # List of (start, end) indices
    context_str: str  # Pre-formatted context
    created_at: float
    ttl: float = 60.0  # Cache TTL in seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

class PrefetchCache:
    """
    LRU cache for pre-computed context spans.
    Stores results keyed by query embedding hash.
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._cache: OrderedDict[str, CachedSpan] = OrderedDict()
    
    def _make_key(self, query: str) -> str:
        """Simple hash for cache key."""
        return str(hash(query.lower().strip()))
    
    def get(self, query: str) -> Optional[CachedSpan]:
        key = self._make_key(query)
        if key in self._cache:
            span = self._cache[key]
            if not span.is_expired():
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return span
            else:
                del self._cache[key]
        return None
    
    def put(self, query: str, spans: List[tuple], context_str: str):
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
        self._cache.clear()

class Prefetcher:
    """
    Async prefetcher that predicts follow-up queries and warms the cache.
    """
    
    def __init__(
        self,
        cache: PrefetchCache,
        compute_fn: Callable[[str], tuple],  # fn(query) -> (spans, context_str)
        predict_fn: Optional[Callable[[str, str], List[str]]] = None  # fn(query, response) -> [predicted_queries]
    ):
        """
        Args:
            cache: The span cache
            compute_fn: Function to compute spans for a query
            predict_fn: Optional function to predict follow-up queries.
                        If None, uses simple heuristics.
        """
        self.cache = cache
        self.compute_fn = compute_fn
        self.predict_fn = predict_fn or self._default_predictions
        self._background_tasks: List[asyncio.Task] = []
    
    def _default_predictions(self, last_query: str, last_response: str) -> List[str]:
        """
        Simple heuristic predictions for follow-up queries.
        In production, replace with a small LLM call.
        """
        predictions = []
        
        # Common follow-up patterns
        patterns = [
            f"What do you mean by {last_query.split()[-1] if last_query.split() else 'that'}?",
            "Can you explain more?",
            "What about the code?",
            "Show me an example",
        ]
        
        # Extract potential entities from response for topic-based predictions
        words = last_response.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 3]
        for word in capitalized[:2]:
            predictions.append(f"Tell me more about {word}")
        
        return predictions[:3]  # Limit to 3 predictions
    
    async def trigger_prefetch(self, last_query: str, last_response: str):
        """
        Called after generating a response. Starts background prefetch.
        """
        predicted = self.predict_fn(last_query, last_response)
        
        # Start background tasks
        for query in predicted:
            if self.cache.get(query) is None:  # Not already cached
                task = asyncio.create_task(self._prefetch_one(query))
                self._background_tasks.append(task)
        
        # Cleanup completed tasks
        self._background_tasks = [t for t in self._background_tasks if not t.done()]
    
    async def _prefetch_one(self, query: str):
        """Compute and cache spans for one predicted query."""
        try:
            # Run compute in executor to not block
            loop = asyncio.get_event_loop()
            spans, context_str = await loop.run_in_executor(None, self.compute_fn, query)
            self.cache.put(query, spans, context_str)
        except Exception as e:
            # Silently fail - prefetch is best-effort
            pass
    
    def get_cached_or_none(self, query: str) -> Optional[str]:
        """
        Check if we have a cached result for this query.
        Returns the pre-formatted context string or None.
        """
        cached = self.cache.get(query)
        return cached.context_str if cached else None
