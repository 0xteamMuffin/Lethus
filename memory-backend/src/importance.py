from typing import List, Dict, Any
from openai import OpenAI
import numpy as np
from .config import settings


class ImportanceDetector:
    """Detects importance of conversation turns"""
    
    # Keywords that indicate important information
    IMPORTANCE_KEYWORDS = [
        "remember", "important", "always", "never", "prefer", "like", "dislike",
        "name is", "i am", "my", "favorite", "hate", "love", "critical",
        "must", "should", "don't", "do not", "every time", "usually"
    ]
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def calculate_importance(self, user_message: str, assistant_message: str) -> float:
        """
        Calculate importance score for a turn (0.0 to 1.0)
        Combines keyword matching and semantic analysis
        """
        # Keyword-based scoring
        keyword_score = self._keyword_importance(user_message)
        
        # Length-based scoring (longer, detailed messages often more important)
        length_score = min(len(user_message.split()) / 50.0, 1.0)
        
        # Combined score
        importance = (keyword_score * 0.6) + (length_score * 0.4)
        
        return min(importance, 1.0)
    
    def _keyword_importance(self, text: str) -> float:
        """Calculate importance based on keyword presence"""
        text_lower = text.lower()
        matches = sum(1 for keyword in self.IMPORTANCE_KEYWORDS if keyword in text_lower)
        
        # Normalize by number of keywords
        return min(matches / 5.0, 1.0)
    
    def should_pin(self, importance_score: float) -> bool:
        """Determine if a turn should be pinned"""
        return importance_score >= settings.importance_threshold


class EmbeddingGenerator:
    """Generates embeddings using OpenAI"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "text-embedding-ada-002"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def generate_query_variants(self, query: str) -> List[str]:
        """Generate query variants for better retrieval"""
        variants = [query]
        
        # Add simple variants
        if len(query.split()) > 3:
            # Add question form if not already a question
            if not query.strip().endswith('?'):
                variants.append(f"What about {query}?")
            
            # Add imperative form
            variants.append(f"Tell me about {query}")
        
        return variants
