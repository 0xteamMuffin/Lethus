"""
DYCP Core - Dynamic Context Pruning using Kadane's Algorithm.
Implements the algorithm from "Dynamic Context Pruning for Long-Form Dialogue" paper.

Key concepts:
- tau (0.6): Gain threshold - shifts z-scores so only significantly above-average turns have positive gain
- theta (1.0): Stopping threshold - terminates span if cumulative gain drops more than theta from peak
"""
import numpy as np
from typing import List, Tuple, Optional

from ..config import settings


class DYCPCore:
    """
    Core DYCP implementation with Kadane's Algorithm and Semantic Decay.
    """
    
    def __init__(
        self,
        tau: float = None,
        theta: float = None,
        decay_lambda: float = None
    ):
        """
        Initialize DYCP Core.
        
        Args:
            tau: Gain threshold (default from config: 0.6)
            theta: Stopping threshold (default from config: 1.0)
            decay_lambda: Time decay factor (default from config: 0.98)
        """
        self.tau = tau or settings.dycp_tau
        self.theta = theta or settings.dycp_theta
        self.decay_lambda = decay_lambda or settings.decay_lambda

    def compute_relevance(
        self,
        query_emb: np.ndarray,
        history_embs: np.ndarray,
        turn_indices: Optional[np.ndarray] = None,
        current_turn: Optional[int] = None,
        apply_decay: bool = True
    ) -> np.ndarray:
        """
        Compute cosine similarity with optional semantic decay.
        
        Args:
            query_emb: Query embedding vector (D,)
            history_embs: Matrix of history embeddings (N x D)
            turn_indices: Array of turn indices for each history entry
            current_turn: Current turn number (for decay calculation)
            apply_decay: Whether to apply semantic decay
            
        Returns:
            Array of relevance scores (N,)
        """
        if len(history_embs) == 0:
            return np.array([])
        
        # Cosine similarity
        norm_query = np.linalg.norm(query_emb)
        norm_history = np.linalg.norm(history_embs, axis=1)
        
        dot_products = np.dot(history_embs, query_emb)
        similarities = dot_products / (norm_history * norm_query + 1e-9)
        
        # Apply Semantic Decay: older turns need higher base similarity to be relevant
        if apply_decay and turn_indices is not None and current_turn is not None:
            ages = current_turn - turn_indices
            decay_factors = np.power(self.decay_lambda, ages)
            similarities = similarities * decay_factors
        
        return similarities

    def get_pruned_indices(self, similarities: np.ndarray) -> List[Tuple[int, int]]:
        """
        KadaneDial: Modified Kadane's algorithm for dynamic context span selection.
        
        From the DYCP paper (Section 5.4):
        - tau: Gain threshold - shifts z-scores so only significantly above-average turns have positive gain
        - theta: Stopping threshold - if cumulative gain drops more than theta from peak, terminate span
        
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

        # 2. Calculate Gain = z_score - tau
        # Positive gain means the turn is more than tau standard deviations above mean
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
                
                # theta-based early stopping: drop from peak exceeds theta
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
            i = max(i, peak_end + 1)
        
        return selected_spans

    def format_context(
        self,
        turns: List[dict],
        spans: List[Tuple[int, int]],
        header: str = "RELEVANT CONTEXT (DYCP)"
    ) -> str:
        """
        Format selected spans into a context string.
        
        Args:
            turns: List of turn dictionaries with 'role' and 'content' keys
            spans: List of (start, end) index tuples
            header: Header text for the context block
            
        Returns:
            Formatted context string
        """
        if not spans:
            return "No relevant context found in memory."
        
        context_str = f"--- {header} ---\n"
        
        prev_end = -1
        for start, end in spans:
            if start > prev_end + 1:
                context_str += "\n[...earlier context omitted...]\n\n"
            
            for i in range(start, end + 1):
                if i < len(turns):
                    t = turns[i]
                    role = t.get("role", "unknown").upper()
                    content = t.get("content", "")
                    context_str += f"[{role}]: {content}\n"
            
            prev_end = end
        
        context_str += "\n--- END CONTEXT ---"
        return context_str
