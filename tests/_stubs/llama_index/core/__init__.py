"""Minimal stubs for llama_index.core used by src.tools in tests."""

from typing import Any, Dict, List, Optional


class Document:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}


class Settings:
    llm = None
    embed_model = None
    chunk_size = 1024
    chunk_overlap = 128


class _VectorStoreData:
    def __init__(self):
        self.embedding_dict: Dict[str, List[float]] = {}


class _VectorStore:
    def __init__(self):
        self._data = _VectorStoreData()


class _QueryResponse:
    def __init__(self, text: str = ""):
        self.response = text
        self.source_nodes: List[Any] = []


class _QueryEngine:
    def __init__(self, index: "VectorStoreIndex"):
        self._index = index

    def query(self, _prompt: str) -> _QueryResponse:
        return _QueryResponse("stub response")


class VectorStoreIndex:
    def __init__(self, docs: Optional[List[Document]] = None):
        self._docs = docs or []
        self.vector_store = _VectorStore()

    @classmethod
    def from_documents(cls, docs: List[Document]) -> "VectorStoreIndex":
        return cls(list(docs))

    def as_query_engine(self, similarity_top_k: int = 3) -> _QueryEngine:
        return _QueryEngine(self)

