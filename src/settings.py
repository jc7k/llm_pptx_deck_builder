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

    # Validation/quality configuration
    strict_validation: bool = Field(
        default=False,
        description="Use strict content validation rules (set True for prod)",
    )

    # HTTP Configuration
    user_agent: Optional[str] = Field(
        default=None, description="User agent string for HTTP requests"
    )
    verify_ssl: bool = Field(
        default=True, description="Verify SSL certificates for HTTP requests"
    )

    # Phoenix Observability Configuration
    enable_phoenix: bool = Field(
        default=True, description="Enable Arize Phoenix observability and tracing"
    )
    phoenix_host: str = Field(
        default="127.0.0.1", description="Phoenix server host"
    )
    phoenix_port: int = Field(
        default=6006, description="Phoenix server port"
    )
    phoenix_collector_endpoint: Optional[str] = Field(
        default=None, 
        description="Phoenix collector endpoint (if using remote Phoenix)"
    )
    phoenix_project_name: str = Field(
        default="llm-pptx-deck-builder", 
        description="Project name for Phoenix tracing"
    )
    phoenix_enable_evaluations: bool = Field(
        default=True, description="Enable Phoenix evaluations for content quality"
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

    def __delattr__(self, name):  # Support mocking frameworks cleaning up
        try:
            # Attempt to delete attribute on the underlying settings if it exists
            if hasattr(get_settings(), name):
                # Reset to default by setting to current value (no-op)
                # Deletion isn't meaningful for BaseSettings; ignore safely.
                return
        except Exception:
            pass


settings = SettingsProxy()
