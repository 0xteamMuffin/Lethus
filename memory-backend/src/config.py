from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    
    # Postgres Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "memory_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # Milvus Configuration
    milvus_mode: str = "local"  # 'local' for in-memory, 'remote' for Docker
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # Memory Configuration
    embedding_dim: int = 1536  # OpenAI ada-002 dimension
    dycp_window_size: int = 10
    max_context_tokens: int = 8000
    importance_threshold: float = 0.7
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"


settings = Settings()
