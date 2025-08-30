"""Dependencies and configuration for the LLM PPTX Deck Builder."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings as LlamaIndexSettings
from .settings import settings

# Phoenix imports (conditional to handle missing dependencies gracefully)
try:
    import phoenix as px
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False


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


def configure_phoenix():
    """Configure Arize Phoenix observability instrumentation."""
    if not PHOENIX_AVAILABLE:
        print("Warning: Phoenix is not available. Install with: uv sync")
        return None
        
    if not settings.enable_phoenix:
        print("Phoenix observability is disabled via configuration")
        return None
    
    # Set Phoenix working directory for consistent data storage
    from pathlib import Path
    phoenix_dir = Path.home() / ".phoenix_llm_pptx"
    phoenix_dir.mkdir(exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(phoenix_dir)
    
    try:
        # Check if Phoenix server is running
        import requests
        try:
            response = requests.get(f"http://{settings.phoenix_host}:{settings.phoenix_port}", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Phoenix server at http://{settings.phoenix_host}:{settings.phoenix_port}")
            else:
                print(f"⚠️  Phoenix server not responding at http://{settings.phoenix_host}:{settings.phoenix_port}")
                print(f"💡 Start persistent server with: uv run python scripts/start_phoenix.py")
        except requests.RequestException:
            print(f"⚠️  Phoenix server not found at http://{settings.phoenix_host}:{settings.phoenix_port}")
            print(f"💡 Start persistent server with: uv run python scripts/start_phoenix.py")

        # Configure OpenTelemetry tracing
        resource = Resource.create({
            "service.name": settings.phoenix_project_name,
            "service.version": "1.0.0",
        })

        if settings.phoenix_collector_endpoint:
            # Use remote Phoenix collector
            exporter = OTLPSpanExporter(
                endpoint=settings.phoenix_collector_endpoint,
                headers={}
            )
        else:
            # Use local Phoenix server
            exporter = OTLPSpanExporter(
                endpoint=f"http://{settings.phoenix_host}:{settings.phoenix_port}/v1/traces",
                headers={}
            )

        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)
        otel_trace.set_tracer_provider(tracer_provider)

        # Initialize instrumentors for automatic tracing
        print("Initializing Phoenix instrumentors...")
        
        # OpenAI instrumentation
        OpenAIInstrumentor().instrument()
        print("✓ OpenAI instrumentation enabled")
        
        # LangChain instrumentation  
        LangChainInstrumentor().instrument()
        print("✓ LangChain instrumentation enabled")
        
        # LlamaIndex instrumentation
        LlamaIndexInstrumentor().instrument()
        print("✓ LlamaIndex instrumentation enabled")
        
        print(f"✓ Phoenix observability initialized successfully!")
        return session
        
    except Exception as e:
        print(f"Warning: Failed to initialize Phoenix: {e}")
        print("Continuing without Phoenix observability...")
        return None


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


# Initialize configurations on module import (can be disabled via env)
if os.environ.get("DECK_BUILDER_AUTOINIT", "1") == "1":
    try:
        validate_api_keys()
        configure_llamaindex_settings()
        configure_langsmith()
        configure_phoenix()
        configure_http_settings()
    except Exception as e:
        print(f"Warning: Failed to initialize dependencies: {e}")
        print("Please check your environment configuration.")
