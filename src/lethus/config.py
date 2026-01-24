"""
Lethus configuration.
Supports both MCP and REST API modes.
"""
from pydantic_settings import BaseSettings
from typing import List, Literal, Optional


class Settings(BaseSettings):
    # === Mode ===
    mode: Literal["mcp", "api", "both"] = "both"
    
    # === API Configuration ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    
    # === PostgreSQL (for REST API conversation storage) ===
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lethus_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # === Milvus Vector Store ===
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "lethus_memory"
    
    # === Embeddings ===
    # Provider: "local" (sentence-transformers) or "openai"
    embedding_provider: Literal["local", "openai"] = "openai"
    # Local models: all-MiniLM-L6-v2 (384d, 80MB), nomic-embed-text-v1.5 (768d, 550MB), BAAI/bge-m3 (1024d, 2.2GB)
    local_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    local_embedding_dim: int = 768
    # OpenAI: text-embedding-3-small (1536d, $0.02/1M), text-embedding-3-large (3072d, $0.13/1M)
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    openai_api_key: Optional[str] = None
    
    # === DYCP Algorithm (from paper Section 5.4) ===
    # Note: Lower tau (0.3) and higher theta (1.5) may improve recall
    dycp_tau: float = 0.6  # Gain threshold - turns need z-score > tau to have positive gain
    dycp_theta: float = 1.0  # Stopping threshold - terminate span if drop from peak > theta
    
    # === Semantic Decay ===
    decay_lambda: float = 0.98  # 2% decay per turn - older turns need higher relevance
    
    # === Ghost Graph ===
    use_spacy: bool = True  # REQUIRED for proper entity extraction
    ghost_graph_boost: float = 1.5  # Multiplicative boost for entity-linked turns
    ghost_graph_additive_boost: float = 0.35  # Additive boost per semantic type match
    
    # === Prefetch Cache ===
    prefetch_enabled: bool = True
    prefetch_cache_size: int = 50
    prefetch_ttl: float = 60.0
    
    # === Context Processing ===
    max_context_tokens: int = 8000
    importance_threshold: float = 0.7
    confidence_threshold: float = 0.3
    
    # === LLM (for REST API) ===
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on provider"""
        if self.embedding_provider == "openai":
            return self.openai_embedding_dim
        return self.local_embedding_dim
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_prefix = "LETHUS_"


settings = Settings()
