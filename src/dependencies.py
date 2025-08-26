"""Dependencies and configuration for the LLM PPTX Deck Builder."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings as LlamaIndexSettings
from .settings import settings


def get_openai_llm(model: Optional[str] = None, temperature: float = 0.1) -> ChatOpenAI:
    """Get configured OpenAI LLM for LangChain."""
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model or settings.openai_model,
        temperature=temperature,
    )


def get_llamaindex_llm(
    model: Optional[str] = None, temperature: float = 0.1
) -> LlamaIndexOpenAI:
    """Get configured OpenAI LLM for LlamaIndex."""
    return LlamaIndexOpenAI(
        api_key=settings.openai_api_key,
        model=model or settings.openai_model,
        temperature=temperature,
    )


def get_embedding_model(model: Optional[str] = None) -> OpenAIEmbedding:
    """Get configured embedding model."""
    return OpenAIEmbedding(
        api_key=settings.openai_api_key, model=model or settings.embedding_model
    )


def configure_llamaindex_settings():
    """Configure global LlamaIndex settings."""
    LlamaIndexSettings.llm = get_llamaindex_llm()
    LlamaIndexSettings.embed_model = get_embedding_model()
    LlamaIndexSettings.chunk_size = settings.chunk_size
    LlamaIndexSettings.chunk_overlap = settings.chunk_overlap


def configure_langsmith():
    """Configure LangSmith tracing if enabled."""
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key


def configure_http_settings():
    """Configure HTTP settings like User Agent."""
    if settings.user_agent:
        os.environ["USER_AGENT"] = settings.user_agent


def get_brave_search_headers() -> dict:
    """Get headers for Brave Search API."""
    return {
        "X-Subscription-Token": settings.brave_api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }


def validate_api_keys():
    """Validate that required API keys are present."""
    errors = []

    if not settings.brave_api_key:
        errors.append("BRAVE_API_KEY environment variable is required")

    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY environment variable is required")

    if errors:
        raise ValueError(f"Missing required API keys: {', '.join(errors)}")


# Initialize configurations on module import
try:
    validate_api_keys()
    configure_llamaindex_settings()
    configure_langsmith()
except Exception as e:
    print(f"Warning: Failed to initialize dependencies: {e}")
    print("Please check your environment configuration.")
