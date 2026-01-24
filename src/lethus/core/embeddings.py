from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..config import settings


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        embedding_dim: int = None
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.base_url = base_url or settings.openai_base_url
        self._embedding_dim = embedding_dim or settings.openai_embedding_dim
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    @property
    def dim(self) -> int:
        return self._embedding_dim
    
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
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_dim: Optional[int] = None
) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        embedding_dim=embedding_dim
    )
