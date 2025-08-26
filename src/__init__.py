"""LLM PPTX Deck Builder package."""

__version__ = "0.1.0"
__author__ = "Claude Code"
__description__ = (
    "LangChain-powered PowerPoint deck builder with Brave Search and LlamaIndex RAG"
)

from .deck_builder_agent import build_deck_sync, build_deck
from .models import DeckBuilderRequest, DeckBuilderResponse
from .settings import settings

__all__ = [
    "build_deck_sync",
    "build_deck",
    "DeckBuilderRequest",
    "DeckBuilderResponse",
    "settings",
]
