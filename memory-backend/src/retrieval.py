from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from .database import Turn
from .milvus_client import milvus_client
from .importance import EmbeddingGenerator
from .config import settings
import numpy as np


class TurnPairer:
    """Pairs user messages with assistant responses"""
    
    @staticmethod
    def create_turn_pair(user_msg: str, assistant_msg: str) -> str:
        """Create a combined representation of a turn"""
        return f"User: {user_msg}\nAssistant: {assistant_msg}"


class RelevanceScorer:
    """Scores relevance of turns to current query"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_gen = embedding_generator
    
    def score_turns(
        self,
        query: str,
        conversation_id: int,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Score and retrieve relevant turns
        Returns list of turns with relevance scores
        """
        # Generate query embedding
        query_embedding = self.embedding_gen.generate_embedding(query)
        
        # Search in Milvus
        similar_turns = milvus_client.search_similar(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=top_k
        )
        
        return similar_turns
    
    def multi_query_retrieval(
        self,
        queries: List[str],
        conversation_id: int,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval with multiple query variants
        Combines and deduplicates results
        """
        all_results = {}
        
        for query in queries:
            results = self.score_turns(query, conversation_id, top_k)
            
            for result in results:
                turn_id = result["turn_id"]
                if turn_id not in all_results:
                    all_results[turn_id] = result
                else:
                    # Keep the higher similarity score
                    if result["similarity"] > all_results[turn_id]["similarity"]:
                        all_results[turn_id] = result
        
        # Sort by similarity
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )
        
        return sorted_results[:top_k]


class DYCPSpanSelector:
    """Selects conversation spans using DYCP (Do You Copy?) algorithm"""
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or settings.dycp_window_size
    
    def select_spans(
        self,
        relevant_turns: List[Dict[str, Any]],
        db: Session,
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Select conversation spans around relevant turns
        """
        if not relevant_turns:
            return []
        
        spans = []
        
        for turn in relevant_turns:
            turn_number = turn["turn_number"]
            
            # Define span window
            start_turn = max(1, turn_number - self.window_size // 2)
            end_turn = turn_number + self.window_size // 2
            
            # Retrieve turns in span from database
            span_turns = db.query(Turn).filter(
                Turn.conversation_id == conversation_id,
                Turn.turn_number >= start_turn,
                Turn.turn_number <= end_turn
            ).order_by(Turn.turn_number).all()
            
            if span_turns:
                spans.append({
                    "center_turn": turn_number,
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "relevance_score": turn["similarity"],
                    "turns": [
                        {
                            "id": t.id,
                            "turn_number": t.turn_number,
                            "user_message": t.user_message,
                            "assistant_message": t.assistant_message,
                            "importance_score": t.importance_score
                        }
                        for t in span_turns
                    ]
                })
        
        return spans
    
    def merge_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping or adjacent spans
        """
        if not spans:
            return []
        
        # Sort by start turn
        sorted_spans = sorted(spans, key=lambda x: x["start_turn"])
        
        merged = [sorted_spans[0]]
        
        for current in sorted_spans[1:]:
            last_merged = merged[-1]
            
            # Check for overlap or adjacency (within 2 turns)
            if current["start_turn"] <= last_merged["end_turn"] + 2:
                # Merge spans
                last_merged["end_turn"] = max(last_merged["end_turn"], current["end_turn"])
                
                # Combine turns and deduplicate
                turn_map = {t["id"]: t for t in last_merged["turns"]}
                for turn in current["turns"]:
                    if turn["id"] not in turn_map:
                        turn_map[turn["id"]] = turn
                
                last_merged["turns"] = sorted(
                    turn_map.values(),
                    key=lambda x: x["turn_number"]
                )
                
                # Update relevance score (max)
                last_merged["relevance_score"] = max(
                    last_merged["relevance_score"],
                    current["relevance_score"]
                )
            else:
                merged.append(current)
        
        return merged
