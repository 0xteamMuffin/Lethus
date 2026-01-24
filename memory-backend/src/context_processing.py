import tiktoken
from typing import List, Dict, Any
from .config import settings


class TokenCounter:
    """Counts tokens using tiktoken"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages"""
        total = 0
        for message in messages:
            # Count tokens for role and content
            total += self.count_tokens(message.get("role", ""))
            total += self.count_tokens(message.get("content", ""))
            total += 4  # Message formatting tokens
        total += 2  # Conversation formatting tokens
        return total


class ContextTrimmer:
    """Trims context to fit within token budget"""
    
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.token_counter = TokenCounter()
    
    def trim_spans(
        self,
        spans: List[Dict[str, Any]],
        pinned_memories: List[Dict[str, Any]],
        system_prompt: str,
        current_message: str
    ) -> List[Dict[str, Any]]:
        """
        Trim spans to fit within token budget
        Priority: system prompt > pinned memories > high-relevance spans > other spans
        """
        # Calculate base tokens (system, current message, pinned)
        base_tokens = (
            self.token_counter.count_tokens(system_prompt) +
            self.token_counter.count_tokens(current_message)
        )
        
        pinned_tokens = sum(
            self.token_counter.count_tokens(m.get("content", ""))
            for m in pinned_memories
        )
        
        available_tokens = self.max_tokens - base_tokens - pinned_tokens - 500  # Reserve for response
        
        if available_tokens <= 0:
            return []
        
        # Sort spans by relevance
        sorted_spans = sorted(
            spans,
            key=lambda x: x["relevance_score"],
            reverse=True
        )
        
        trimmed_spans = []
        current_tokens = 0
        
        for span in sorted_spans:
            # Calculate span tokens
            span_text = self._span_to_text(span)
            span_tokens = self.token_counter.count_tokens(span_text)
            
            if current_tokens + span_tokens <= available_tokens:
                trimmed_spans.append(span)
                current_tokens += span_tokens
            else:
                # Try to fit partial span
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 200:  # Minimum useful span size
                    partial_span = self._trim_span_to_tokens(span, remaining_tokens)
                    if partial_span:
                        trimmed_spans.append(partial_span)
                break
        
        return trimmed_spans
    
    def _span_to_text(self, span: Dict[str, Any]) -> str:
        """Convert span to text for token counting"""
        text_parts = []
        for turn in span["turns"]:
            text_parts.append(f"User: {turn['user_message']}")
            text_parts.append(f"Assistant: {turn['assistant_message']}")
        return "\n".join(text_parts)
    
    def _trim_span_to_tokens(
        self,
        span: Dict[str, Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Trim a span to fit within token limit"""
        trimmed_turns = []
        current_tokens = 0
        
        # Keep turns from center outward
        center_idx = len(span["turns"]) // 2
        indices = []
        
        # Generate indices from center outward
        for i in range(len(span["turns"])):
            if i % 2 == 0:
                idx = center_idx + i // 2
            else:
                idx = center_idx - (i + 1) // 2
            
            if 0 <= idx < len(span["turns"]):
                indices.append(idx)
        
        for idx in indices:
            turn = span["turns"][idx]
            turn_text = f"User: {turn['user_message']}\nAssistant: {turn['assistant_message']}"
            turn_tokens = self.token_counter.count_tokens(turn_text)
            
            if current_tokens + turn_tokens <= max_tokens:
                trimmed_turns.append((idx, turn))
                current_tokens += turn_tokens
            else:
                break
        
        if not trimmed_turns:
            return None
        
        # Sort back to original order
        trimmed_turns.sort(key=lambda x: x[0])
        
        return {
            **span,
            "turns": [t[1] for t in trimmed_turns],
            "trimmed": True
        }


class ConfidenceChecker:
    """Checks confidence in retrieved context and provides fallback"""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
    
    def check_confidence(
        self,
        spans: List[Dict[str, Any]],
        pinned_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if retrieved context is confident enough
        Returns confidence score and fallback strategy
        """
        if not spans and not pinned_memories:
            return {
                "confident": False,
                "score": 0.0,
                "fallback": "recent",
                "reason": "No relevant context found"
            }
        
        # Calculate confidence based on relevance scores and coverage
        if spans:
            avg_relevance = sum(s["relevance_score"] for s in spans) / len(spans)
            max_relevance = max(s["relevance_score"] for s in spans)
            
            # Confidence increases with both high average and high max
            confidence = (avg_relevance * 0.4) + (max_relevance * 0.6)
        else:
            confidence = 0.5  # Moderate confidence if we have pinned memories
        
        # Boost confidence if we have pinned memories
        if pinned_memories:
            confidence = min(confidence + 0.2, 1.0)
        
        is_confident = confidence >= self.confidence_threshold
        
        return {
            "confident": is_confident,
            "score": confidence,
            "fallback": None if is_confident else "recent",
            "reason": None if is_confident else "Low relevance scores"
        }
    
    def get_fallback_context(
        self,
        db,
        conversation_id: int,
        num_turns: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent conversation turns as fallback"""
        from .database import Turn
        
        recent_turns = db.query(Turn).filter(
            Turn.conversation_id == conversation_id
        ).order_by(Turn.turn_number.desc()).limit(num_turns).all()
        
        return [
            {
                "id": t.id,
                "turn_number": t.turn_number,
                "user_message": t.user_message,
                "assistant_message": t.assistant_message,
                "importance_score": t.importance_score
            }
            for t in reversed(recent_turns)
        ]
