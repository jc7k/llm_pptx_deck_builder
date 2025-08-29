"""Stub WebBaseLoader used by tests (patched in tests)."""

from typing import Any, Dict, List, Optional


class _Doc:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = text
        self.metadata = metadata or {}


class WebBaseLoader:
    def __init__(self, url: str, requests_kwargs: Optional[Dict[str, Any]] = None):
        self.url = url
        self.requests_kwargs = requests_kwargs or {}

    def load(self) -> List[_Doc]:
        return []

