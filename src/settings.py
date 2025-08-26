"""Environment configuration for the LLM PPTX Deck Builder."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    brave_api_key: str = Field(..., description="Brave Search API key")
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Optional: LangSmith Tracing
    langchain_tracing_v2: bool = Field(
        default=False, description="Enable LangSmith tracing"
    )
    langchain_api_key: Optional[str] = Field(
        default=None, description="LangSmith API key"
    )

    # Model Configuration
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )

    # Search Configuration
    max_search_results: int = Field(
        default=15, description="Maximum search results to process"
    )
    max_documents: int = Field(default=20, description="Maximum documents to load")

    # RAG Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size for indexing")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    similarity_top_k: int = Field(
        default=10, description="Top K results for similarity search"
    )

    # Output Configuration
    default_output_dir: str = Field(
        default="output", description="Default output directory"
    )
    
    # HTTP Configuration
    user_agent: Optional[str] = Field(
        default=None, description="User agent string for HTTP requests"
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


# Global settings instance - lazy loaded
_settings = None

def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# For backward compatibility, create a property-like access
class SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)
    
    def __setattr__(self, name, value):
        setattr(get_settings(), name, value)

settings = SettingsProxy()
