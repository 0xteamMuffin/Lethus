import numpy as np
from typing import List, Tuple, Optional

from ..config import settings


class DYCPCore:
    def __init__(
        self,
        tau: float = None,
        theta: float = None,
        decay_lambda: float = None
    ):
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
        if len(history_embs) == 0:
            return np.array([])
        
        norm_query = np.linalg.norm(query_emb)
        norm_history = np.linalg.norm(history_embs, axis=1)
        
        dot_products = np.dot(history_embs, query_emb)
        similarities = dot_products / (norm_history * norm_query + 1e-9)
        
        if apply_decay and turn_indices is not None and current_turn is not None:
            ages = current_turn - turn_indices
            decay_factors = np.power(self.decay_lambda, ages)
            similarities = similarities * decay_factors
        
        return similarities

    def get_pruned_indices(self, similarities: np.ndarray) -> List[Tuple[int, int]]:
        if len(similarities) == 0:
            return []

        std = np.std(similarities)
        if std < 1e-9:
            return []
        
        mean = np.mean(similarities)
        z_scores = (similarities - mean) / std
        gains = z_scores - self.tau

        selected_spans = []
        n = len(gains)
        i = 0
        
        while i < n:
            while i < n and gains[i] <= 0:
                i += 1
            
            if i >= n:
                break
            
            span_start = i
            cumulative = 0.0
            peak_cumulative = 0.0
            peak_end = i
            
            while i < n:
                cumulative += gains[i]
                
                if cumulative > peak_cumulative:
                    peak_cumulative = cumulative
                    peak_end = i
                
                drop_from_peak = peak_cumulative - cumulative
                if drop_from_peak > self.theta:
                    break
                
                if cumulative <= 0:
                    break
                
                i += 1
            
            if peak_cumulative > 0:
                selected_spans.append((span_start, peak_end))
            
            i = max(i, peak_end + 1)
        
        return selected_spans

    def format_context(
        self,
        turns: List[dict],
        spans: List[Tuple[int, int]],
        header: str = "RELEVANT CONTEXT (DYCP)"
    ) -> str:
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
