"""Pytest configuration and fixtures for testing."""

# Ensure test-only stubs are available on import path before other imports
import sys
from pathlib import Path
_STUBS_DIR = Path(__file__).resolve().parent / "_stubs"
if _STUBS_DIR.exists():
    sys.path.insert(0, str(_STUBS_DIR))

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage


@pytest.fixture
def fake_llm():
    """Fixture for FakeListLLM testing."""
    responses = [
        '{"topic": "AI in Education", "objective": "Overview of AI applications in education", "slide_titles": ["Title", "Agenda", "Introduction", "Applications", "Benefits", "Challenges", "Future", "Conclusion"], "target_audience": "Educators", "duration_minutes": 15}',
        '{"title": "Introduction", "bullets": ["AI transforms education", "Personalized learning", "Intelligent tutoring"], "speaker_notes": "AI is revolutionizing education through personalized learning experiences [1]", "slide_type": "content"}'
    ]
    return FakeListLLM(responses=responses)


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return [
        {
            "url": "https://example.com/ai-education",
            "title": "AI in Education Overview",
            "snippet": "Artificial intelligence is transforming education...",
            "published_date": "2024-01-15"
        },
        {
            "url": "https://example.com/ai-trends", 
            "title": "AI Education Trends",
            "snippet": "Latest trends in AI for education...",
            "published_date": "2024-01-20"
        }
    ]


@pytest.fixture
def mock_documents():
    """Mock document data for testing."""
    return [
        {
            "url": "https://example.com/ai-education",
            "title": "AI in Education Overview", 
            "content": "Artificial intelligence is transforming education through personalized learning, intelligent tutoring systems, and automated assessment tools.",
            "metadata": {"source": "https://example.com/ai-education", "title": "AI in Education Overview"}
        },
        {
            "url": "https://example.com/ai-trends",
            "title": "AI Education Trends",
            "content": "The latest trends in AI for education include adaptive learning platforms, virtual teaching assistants, and predictive analytics for student performance.",
            "metadata": {"source": "https://example.com/ai-trends", "title": "AI Education Trends"}
        }
    ]


@pytest.fixture
def mock_vector_index():
    """Mock vector index metadata for testing."""
    return {
        "document_count": 2,
        "chunk_count": 10,
        "embedding_model": "text-embedding-3-small",
        "created_at": "2024-01-25T10:00:00",
        "index_id": "test_index_123",
        "_index": Mock()  # Mock LlamaIndex object
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_api_keys():
    """Mock API keys environment variables."""
    with patch.dict(os.environ, {
        'BRAVE_API_KEY': 'test_brave_key',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        yield


@pytest.fixture
def sample_presentation_outline():
    """Sample presentation outline for testing."""
    return {
        "topic": "AI in Education",
        "objective": "Comprehensive overview of AI applications in education",
        "slide_titles": [
            "Title Slide",
            "Agenda", 
            "Introduction",
            "AI Applications",
            "Benefits",
            "Challenges",
            "Future Outlook",
            "Conclusion"
        ],
        "target_audience": "Educators and administrators",
        "duration_minutes": 15
    }


@pytest.fixture 
def sample_slide_specs():
    """Sample slide specifications for testing."""
    return [
        {
            "title": "Introduction",
            "bullets": [
                "AI is transforming education",
                "Personalized learning experiences", 
                "Intelligent tutoring systems"
            ],
            "speaker_notes": "Artificial intelligence is revolutionizing education through personalized learning experiences and intelligent systems [1]",
            "slide_type": "content",
            "references": ["https://example.com/ai-education"]
        },
        {
            "title": "Applications",
            "bullets": [
                "Adaptive learning platforms",
                "Automated assessment tools",
                "Virtual teaching assistants"
            ],
            "speaker_notes": "Key applications include adaptive learning platforms that adjust to student needs [2]",
            "slide_type": "content", 
            "references": ["https://example.com/ai-trends"]
        }
    ]
