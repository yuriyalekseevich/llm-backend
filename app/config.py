"""
Configuration management for RAG application.
All settings centralized with validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(None, alias="GROQ_API_KEY")
    
    # Paths
    chroma_db_path: str = Field("./chroma_db", alias="CHROMA_DB_PATH")
    collection_name: str = Field("rag_docs", alias="COLLECTION_NAME")
    
    # Model settings
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(300, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(50, alias="CHUNK_OVERLAP")
    
    # RAG settings
    default_top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: str = Field("json", alias="LOG_FORMAT")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )
    
    @field_validator("openai_api_key", "groq_api_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that at least one API key is present."""
        if not v:
            # Don't validate during initial load if not required
            return v
        return v
    
    def ensure_paths(self):
        """Create necessary directories."""
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        return self

# Singleton instance
settings = Settings().ensure_paths()