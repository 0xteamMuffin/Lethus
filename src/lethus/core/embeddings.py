"""
Embedding providers: Local (sentence-transformers) or OpenAI.
Abstracts embedding generation to support multiple backends.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..config import settings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.local_embedding_model
        self._model = None
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # nomic models require trust_remote_code
            trust_remote = "nomic" in self.model_name.lower()
            self._model = SentenceTransformer(self.model_name, trust_remote_code=trust_remote)
        return self._model
    
    @property
    def dim(self) -> int:
        return settings.local_embedding_dim
    
    def embed_text(self, text: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small/large)."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self._client = None
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    @property
    def dim(self) -> int:
        return settings.openai_embedding_dim
    
    def embed_text(self, text: str) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=text, model=self.model)
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model)
        embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
        return np.stack(embeddings)


def get_embedding_provider(
    provider: str = None,
    api_key: Optional[str] = None
) -> EmbeddingProvider:
    """
    Factory function to get the configured embedding provider.
    
    Args:
        provider: "local" or "openai" (defaults to config setting)
        api_key: OpenAI API key (only needed for openai provider)
    """
    provider = provider or settings.embedding_provider
    
    if provider == "openai":
        return OpenAIEmbeddingProvider(api_key=api_key)
    else:
        return LocalEmbeddingProvider()
