"""Pydantic models for the LLM PPTX Deck Builder."""

from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage


class DeckBuilderState(TypedDict):
    """State schema for the LangGraph workflow."""

    messages: List[AnyMessage]
    user_request: str
    search_results: List[Dict]
    documents: List[Dict]  # LangChain Document objects
    vector_index: Optional[Dict]  # LlamaIndex metadata
    outline: Optional[Dict]
    slide_specs: List[Dict]
    references: List[str]
    template_path: Optional[str]
    output_path: str
    status: str


class SearchResult(BaseModel):
    """Model for search results from Brave Search API."""

    url: str = Field(..., description="URL of the search result")
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Snippet/description of the search result")
    published_date: Optional[str] = Field(
        None, description="Publication date if available"
    )


class WebDocument(BaseModel):
    """Model for web documents loaded by LangChain."""

    url: str = Field(..., description="Source URL")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Extracted text content")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class PresentationOutline(BaseModel):
    """Model for presentation outline."""

    topic: str = Field(..., description="Main presentation topic")
    objective: str = Field(..., description="Presentation objective/purpose")
    slide_titles: List[str] = Field(..., description="List of slide titles")
    target_audience: Optional[str] = Field(None, description="Target audience")
    duration_minutes: Optional[int] = Field(None, description="Expected duration")


class SlideContent(BaseModel):
    """Model for individual slide content."""

    title: str = Field(..., description="Slide title")
    bullets: List[str] = Field(..., description="Bullet points for the slide")
    speaker_notes: str = Field(..., description="Speaker notes with citations")
    slide_type: str = Field(default="content", description="Type of slide")
    references: List[str] = Field(
        default_factory=list, description="Reference URLs for this slide"
    )


class PresentationSpec(BaseModel):
    """Complete presentation specification."""

    outline: PresentationOutline = Field(..., description="Presentation outline")
    slides: List[SlideContent] = Field(..., description="List of slide content")
    references: List[str] = Field(..., description="All unique references")


class DeckBuilderRequest(BaseModel):
    """Request model for deck building."""

    topic: str = Field(..., description="Presentation topic")
    audience: Optional[str] = Field(None, description="Target audience")
    duration: Optional[int] = Field(None, description="Duration in minutes")
    template_path: Optional[str] = Field(
        None, description="Path to PowerPoint template"
    )
    output_path: Optional[str] = Field(None, description="Output file path")
    max_slides: int = Field(default=12, description="Maximum number of slides")


class DeckBuilderResponse(BaseModel):
    """Response model for deck building."""

    success: bool = Field(..., description="Whether the operation was successful")
    output_path: Optional[str] = Field(
        None, description="Path to generated presentation"
    )
    slide_count: Optional[int] = Field(None, description="Number of slides generated")
    references_count: Optional[int] = Field(
        None, description="Number of references included"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class VectorIndexMetadata(BaseModel):
    """Metadata for vector index operations."""

    document_count: int = Field(..., description="Number of documents indexed")
    chunk_count: int = Field(..., description="Number of text chunks")
    embedding_model: str = Field(..., description="Embedding model used")
    created_at: str = Field(..., description="Creation timestamp")
    index_id: Optional[str] = Field(None, description="Index identifier")


class RAGQuery(BaseModel):
    """Model for RAG query operations."""

    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of results to retrieve")
    similarity_threshold: Optional[float] = Field(
        None, description="Minimum similarity threshold"
    )


class RAGResponse(BaseModel):
    """Response model for RAG queries."""

    results: List[Dict] = Field(
        ..., description="Retrieved results with content and metadata"
    )
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")
