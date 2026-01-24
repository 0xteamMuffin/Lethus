"""Database connection managers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from contextlib import contextmanager
from typing import Generator
import logging

from .config import settings

logger = logging.getLogger(__name__)


# PostgreSQL Setup
engine = create_engine(settings.postgres_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[SQLSession, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_postgres():
    """Initialize PostgreSQL database."""
    from .models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("PostgreSQL database initialized")


# Milvus Setup
def init_milvus():
    """Initialize Milvus connection and collections."""
    try:
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        logger.info("Connected to Milvus")
        
        # Create collection for turn embeddings
        collection_name = "turn_embeddings"
        
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="turn_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dimension),
            FieldSchema(name="turn_index", dtype=DataType.INT64),
            FieldSchema(name="is_pinned", dtype=DataType.BOOL),
        ]
        
        schema = CollectionSchema(fields=fields, description="Turn embeddings for conversation memory")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created Milvus collection: {collection_name}")
        
        return collection
        
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        raise


def get_milvus_collection():
    """Get Milvus collection for turn embeddings."""
    return Collection("turn_embeddings")
