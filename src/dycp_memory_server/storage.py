"""
Production Vector Storage using Milvus Cluster.
Supports: Vector search with Time Decay filtering.
"""
from pymilvus import MilvusClient, DataType
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time

class MilvusStorage:
    COLLECTION_NAME = "conversation_memory"
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self, uri: str = "http://localhost:19530"):
        """
        Initialize Milvus connection.
        
        Args:
            uri: Milvus endpoint (e.g., "http://localhost:19530")
        """
        self.client = MilvusClient(uri=uri)
        self._init_collection()

    def _init_collection(self):
        """Create collection with schema if it doesn't exist."""
        if self.client.has_collection(self.COLLECTION_NAME):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM)
        schema.add_field(field_name="turn_index", datatype=DataType.INT64)  # For ordering and decay
        schema.add_field(field_name="timestamp", datatype=DataType.DOUBLE)  # Unix timestamp
        # Entity links for Ghost Graph
        schema.add_field(field_name="entities", datatype=DataType.VARCHAR, max_length=4096)  # JSON string

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def get_next_turn_index(self) -> int:
        """Get the next turn index for ordering."""
        results = self.client.query(
            collection_name=self.COLLECTION_NAME,
            filter="",
            output_fields=["turn_index"],
            limit=1,
            # Sort by turn_index descending to get the max
        )
        if not results:
            return 0
        # Query doesn't support ordering in Milvus Lite the same way, so we do a workaround
        # For production, you'd use a separate counter or MAX aggregation
        all_results = self.client.query(
            collection_name=self.COLLECTION_NAME,
            filter="turn_index >= 0",
            output_fields=["turn_index"]
        )
        if not all_results:
            return 0
        max_idx = max(r["turn_index"] for r in all_results)
        return max_idx + 1

    def add_turn(self, role: str, content: str, embedding: np.ndarray, entities: str = "[]"):
        """
        Add a conversation turn to the collection.
        
        Args:
            role: "user" or "assistant"
            content: Message text
            embedding: Vector embedding (numpy array)
            entities: JSON string of extracted entities e.g. '["API Key", "AWS"]'
        """
        turn_index = self.get_next_turn_index()
        
        data = [{
            "role": role,
            "content": content,
            "embedding": embedding.tolist(),
            "turn_index": turn_index,
            "timestamp": time.time(),
            "entities": entities
        }]
        
        self.client.insert(collection_name=self.COLLECTION_NAME, data=data)

    def search_with_decay(
        self,
        query_embedding: np.ndarray,
        current_turn: int,
        decay_lambda: float = 0.98,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for similar turns with time decay applied.
        
        The decay formula: adjusted_score = similarity * (lambda ^ (current_turn - turn_k))
        
        Args:
            query_embedding: The query vector
            current_turn: The current turn index (for decay calculation)
            decay_lambda: Decay factor (0.98 means 2% decay per turn)
            top_k: Number of candidates to retrieve before decay adjustment
        """
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            data=[query_embedding.tolist()],
            anns_field="embedding",
            limit=top_k,
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities"]
        )

        if not results or not results[0]:
            return []

        # Apply time decay to scores
        adjusted_results = []
        for hit in results[0]:
            turn_idx = hit["entity"]["turn_index"]
            age = current_turn - turn_idx
            decay_factor = decay_lambda ** age
            adjusted_score = hit["distance"] * decay_factor  # Milvus COSINE returns similarity
            
            adjusted_results.append({
                "id": hit["entity"]["id"],
                "role": hit["entity"]["role"],
                "content": hit["entity"]["content"],
                "turn_index": turn_idx,
                "timestamp": hit["entity"]["timestamp"],
                "entities": hit["entity"]["entities"],
                "raw_score": hit["distance"],
                "decayed_score": adjusted_score
            })

        # Re-rank by decayed score
        adjusted_results.sort(key=lambda x: x["decayed_score"], reverse=True)
        return adjusted_results

    def search_by_entity(self, entity_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for turns that mention a specific entity (Ghost Graph lookup).
        Uses string matching on the entities field.
        """
        # Milvus string matching - escape quotes
        safe_entity = entity_name.replace('"', '\\"')
        filter_expr = f'entities like "%{safe_entity}%"'
        
        results = self.client.query(
            collection_name=self.COLLECTION_NAME,
            filter=filter_expr,
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities"],
            limit=limit
        )
        return results

    def get_all_turns_ordered(self) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Get all turns in chronological order.
        Returns turns and their embeddings.
        """
        results = self.client.query(
            collection_name=self.COLLECTION_NAME,
            filter="turn_index >= 0",
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities", "embedding"]
        )
        
        if not results:
            return [], np.array([])

        # Sort by turn_index
        results.sort(key=lambda x: x["turn_index"])
        
        turns = []
        embeddings = []
        for r in results:
            turns.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "turn_index": r["turn_index"],
                "timestamp": r["timestamp"],
                "entities": r["entities"]
            })
            embeddings.append(np.array(r["embedding"], dtype=np.float32))

        return turns, np.stack(embeddings) if embeddings else np.array([])

    def get_current_turn_count(self) -> int:
        """Get total number of turns."""
        results = self.client.query(
            collection_name=self.COLLECTION_NAME,
            filter="turn_index >= 0",
            output_fields=["turn_index"]
        )
        return len(results)

    def clear_history(self):
        """Drop and recreate the collection."""
        self.client.drop_collection(self.COLLECTION_NAME)
        self._init_collection()
