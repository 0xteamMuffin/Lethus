from milvus import default_server
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import List, Dict, Any
import numpy as np
from .config import settings


class MilvusClient:
    def __init__(self):
        self.mode = settings.milvus_mode
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.collection_name = "conversation_embeddings"
        self.collection = None
        self.server = None
        
    def connect(self):
        """Connect to Milvus (local or remote)"""
        if self.mode == "local":
            # Start Milvus Lite (in-memory)
            self.server = default_server
            self.server.start()
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port=self.server.listen_port
            )
        else:
            # Connect to remote Milvus (Docker)
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get collection"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            # Create collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="turn_id", dtype=DataType.INT64),
                FieldSchema(name="conversation_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="turn_number", dtype=DataType.INT64),
            ]
            
            schema = CollectionSchema(fields=fields, description="Conversation turn embeddings")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        
        self.collection.load()
    
    def insert_embedding(
        self,
        turn_id: int,
        conversation_id: int,
        embedding: List[float],
        text: str,
        turn_number: int
    ):
        """Insert a turn embedding into Milvus"""
        data = [
            [turn_id],
            [conversation_id],
            [embedding],
            [text],
            [turn_number]
        ]
        
        self.collection.insert(data)
        self.collection.flush()
    
    def search_similar(
        self,
        query_embedding: List[float],
        conversation_id: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar turns in a conversation"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f"conversation_id == {conversation_id}",
            output_fields=["turn_id", "text", "turn_number"]
        )
        
        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                hits.append({
                    "turn_id": hit.entity.get("turn_id"),
                    "text": hit.entity.get("text"),
                    "turn_number": hit.entity.get("turn_number"),
                    "similarity": hit.score
                })
        
        return hits
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        if self.mode == "local" and self.server:
            self.server.stop()


# Singleton instance
milvus_client = MilvusClient()
