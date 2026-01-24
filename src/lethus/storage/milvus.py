"""
Milvus vector storage for embeddings.
Supports multi-tenant conversations with entity metadata for Ghost Graph.
"""
from pymilvus import MilvusClient, DataType
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time

from ..config import settings


class MilvusStorage:
    """
    Milvus vector storage with support for:
    - Multi-tenant conversations
    - Entity metadata for Ghost Graph
    - Time-based decay queries
    """
    
    def __init__(
        self,
        uri: str = None,
        collection_name: str = None,
        embedding_dim: int = None
    ):
        self.uri = uri or settings.milvus_uri
        self.collection_name = collection_name or settings.milvus_collection
        self.embedding_dim = embedding_dim or settings.embedding_dim
        
        self.client = MilvusClient(uri=self.uri)
        self._init_collection()
    
    def _init_collection(self):
        """Create collection with schema if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            return
        
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="conversation_id", datatype=DataType.INT64)
        schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field(field_name="turn_index", datatype=DataType.INT64)
        schema.add_field(field_name="timestamp", datatype=DataType.DOUBLE)
        schema.add_field(field_name="entities", datatype=DataType.VARCHAR, max_length=4096)  # JSON string
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
    
    def add_turn(
        self,
        role: str,
        content: str,
        embedding: np.ndarray,
        conversation_id: int = 0,
        entities: str = "[]"
    ) -> int:
        """
        Add a conversation turn to the collection.
        
        Returns:
            The turn index
        """
        turn_index = self.get_turn_count(conversation_id)
        
        data = [{
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "embedding": embedding.tolist(),
            "turn_index": turn_index,
            "timestamp": time.time(),
            "entities": entities
        }]
        
        self.client.insert(collection_name=self.collection_name, data=data)
        return turn_index
    
    def get_turn_count(self, conversation_id: int = 0) -> int:
        """Get the number of turns in a conversation."""
        filter_expr = f"conversation_id == {conversation_id}"
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["turn_index"]
        )
        
        if not results:
            return 0
        
        return max(r["turn_index"] for r in results) + 1
    
    def get_all_turns_ordered(
        self,
        conversation_id: int = 0
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Get all turns in chronological order for DYCP algorithm.
        
        Returns:
            (turns, embeddings) tuple
        """
        filter_expr = f"conversation_id == {conversation_id}" if conversation_id else "turn_index >= 0"
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities", "embedding"]
        )
        
        if not results:
            return [], np.array([])
        
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
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        conversation_id: int = 0,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for similar turns using vector similarity.
        """
        filter_expr = f"conversation_id == {conversation_id}" if conversation_id else None
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            anns_field="embedding",
            limit=top_k,
            filter=filter_expr,
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities"]
        )
        
        if not results or not results[0]:
            return []
        
        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit["entity"]["id"],
                "role": hit["entity"]["role"],
                "content": hit["entity"]["content"],
                "turn_index": hit["entity"]["turn_index"],
                "timestamp": hit["entity"]["timestamp"],
                "entities": hit["entity"]["entities"],
                "similarity": hit["distance"]
            })
        
        return hits
    
    def search_by_entity(
        self,
        entity_name: str,
        conversation_id: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for turns mentioning a specific entity.
        """
        safe_entity = entity_name.replace('"', '\\"')
        filter_expr = f'entities like "%{safe_entity}%"'
        
        if conversation_id:
            filter_expr = f'({filter_expr}) and (conversation_id == {conversation_id})'
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["id", "role", "content", "turn_index", "timestamp", "entities"],
            limit=limit
        )
        
        return results
    
    def clear_conversation(self, conversation_id: int):
        """Delete all turns for a conversation."""
        filter_expr = f"conversation_id == {conversation_id}"
        self.client.delete(collection_name=self.collection_name, filter=filter_expr)
    
    def clear_all(self):
        """Drop and recreate the collection."""
        self.client.drop_collection(self.collection_name)
        self._init_collection()


# Default instance
_milvus_storage = None


def get_milvus_storage() -> MilvusStorage:
    """Get or create the default Milvus storage instance"""
    global _milvus_storage
    if _milvus_storage is None:
        _milvus_storage = MilvusStorage()
    return _milvus_storage
