"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lethus_memory"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Memory System Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    max_context_tokens: int = 8000
    importance_threshold: float = 0.6
    confidence_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


settings = Settings()
