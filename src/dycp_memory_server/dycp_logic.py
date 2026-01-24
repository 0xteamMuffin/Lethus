import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

class DYCPCore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", decay_lambda: float = 0.98):
        """
        Initialize the core with a local embedding model.
        
        Args:
            model_name: HuggingFace model for embeddings
            decay_lambda: Time decay factor (0.98 = 2% decay per turn)
        """
        self.model = SentenceTransformer(model_name)
        self.tau = 0.6  # Gain threshold from paper
        self.theta = 1.0  # Stopping threshold from paper
        self.decay_lambda = decay_lambda

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_relevance(
        self,
        query_emb: np.ndarray,
        history_embs: np.ndarray,
        turn_indices: Optional[np.ndarray] = None,
        current_turn: Optional[int] = None,
        apply_decay: bool = True
    ) -> np.ndarray:
        """
        Compute cosine similarity with optional time decay.
        
        Args:
            query_emb: Query embedding vector
            history_embs: Matrix of history embeddings (N x D)
            turn_indices: Array of turn indices for each history entry
            current_turn: Current turn number (for decay calculation)
            apply_decay: Whether to apply semantic decay
        """
        if len(history_embs) == 0:
            return np.array([])
        
        # Cosine similarity
        norm_query = np.linalg.norm(query_emb)
        norm_history = np.linalg.norm(history_embs, axis=1)
        
        dot_products = np.dot(history_embs, query_emb)
        similarities = dot_products / (norm_history * norm_query + 1e-9)
        
        # Apply Semantic Decay if enabled
        if apply_decay and turn_indices is not None and current_turn is not None:
            ages = current_turn - turn_indices
            decay_factors = np.power(self.decay_lambda, ages)
            similarities = similarities * decay_factors
        
        return similarities

    def get_pruned_indices(self, similarities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Modified Kadane's algorithm for dynamic context span selection.
        
        Parameters:
        - τ (tau) = 0.6: Gain threshold - shifts z-scores so only significantly 
          above-average turns have positive gain
        - θ (theta) = 1.0: Stopping threshold - if cumulative gain drops more than 
          theta from peak, terminate the span at the peak position
        
        The algorithm finds MULTIPLE contiguous spans where cumulative gain is positive,
        using theta-based early stopping to avoid including trailing low-relevance turns.
        
        Returns:
            List of (start_idx, end_idx) tuples representing selected spans (inclusive)
        """
        if len(similarities) == 0:
            return []

        # 1. Z-score Normalization
        std = np.std(similarities)
        if std < 1e-9:
            # No variance means no signal - can't distinguish relevant from irrelevant
            return []
        
        mean = np.mean(similarities)
        z_scores = (similarities - mean) / std

        # 2. Calculate Gain = z_score - τ
        # Positive gain means the turn is more than τ standard deviations above mean
        gains = z_scores - self.tau

        # 3. KadaneDial: Find spans with positive cumulative gain
        selected_spans = []
        n = len(gains)
        i = 0
        
        while i < n:
            # Skip negative-gain turns until we find a potential span start
            while i < n and gains[i] <= 0:
                i += 1
            
            if i >= n:
                break
            
            # Start new span at index i
            span_start = i
            cumulative = 0.0
            peak_cumulative = 0.0
            peak_end = i
            
            # Extend span while conditions hold
            while i < n:
                cumulative += gains[i]
                
                # Track the peak cumulative gain and its position
                if cumulative > peak_cumulative:
                    peak_cumulative = cumulative
                    peak_end = i
                
                # θ-based early stopping: drop from peak exceeds theta
                drop_from_peak = peak_cumulative - cumulative
                if drop_from_peak > self.theta:
                    break
                
                # Classical Kadane reset: cumulative went non-positive
                if cumulative <= 0:
                    break
                
                i += 1
            
            # Record span if it captured positive cumulative gain
            if peak_cumulative > 0:
                selected_spans.append((span_start, peak_end))
            
            # Continue scanning from after the span end
            # Use max to ensure forward progress even if we broke early
            i = max(i, peak_end + 1)
        
        return selected_spans

