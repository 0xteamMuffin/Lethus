from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    api_host: str
    api_port: int
    cors_origins: str
    
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    
    milvus_uri: str
    milvus_collection: str
    
    openai_api_key: Optional[str] = None
    openai_base_url: str
    openai_embedding_model: str
    openai_embedding_dim: int
    
    dycp_tau: float
    dycp_theta: float
    decay_lambda: float
    
    use_spacy: bool
    ghost_graph_boost: float
    ghost_graph_additive_boost: float
    
    prefetch_enabled: bool
    prefetch_cache_size: int
    prefetch_ttl: float
    
    max_context_tokens: int
    importance_threshold: float
    confidence_threshold: float
    
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    
    @property
    def embedding_dim(self) -> int:
        return self.openai_embedding_dim
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = str(_ENV_FILE)
        env_prefix = "LETHUS_"


settings = Settings()
