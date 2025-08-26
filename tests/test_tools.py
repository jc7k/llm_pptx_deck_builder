"""Unit tests for tools module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import os
from src.tools import (
    search_web,
    load_web_documents, 
    extract_urls_from_search_results,
    create_vector_index,
    generate_outline,
    generate_slide_content,
    create_presentation,
    deduplicate_references
)


class TestSearchWeb:
    """Test cases for search_web tool."""
    
    @patch('src.tools.requests.get')
    @patch('src.tools.get_brave_search_headers')
    def test_search_web_success(self, mock_headers, mock_get, mock_api_keys):
        """Test successful web search."""
        mock_headers.return_value = {"Authorization": "Bearer test_key"}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/test",
                        "title": "Test Article",
                        "description": "Test description",
                        "age": "2024-01-15"
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_web.invoke({"query": "test query", "count": 5})
        
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/test"
        assert result[0]["title"] == "Test Article"
    
    @patch('src.tools.requests.get')
    def test_search_web_failure(self, mock_get, mock_api_keys):
        """Test search web with API failure."""
        mock_get.side_effect = Exception("API Error")
        
        result = search_web.invoke({"query": "test query", "count": 5})
        
        assert result == []


class TestLoadWebDocuments:
    """Test cases for load_web_documents tool."""
    
    @patch('src.tools.WebBaseLoader')
    def test_load_web_documents_success(self, mock_loader_class):
        """Test successful document loading."""
        mock_loader = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"title": "Test Title", "source": "https://example.com"}
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        urls = ["https://example.com/test"]
        result = load_web_documents.invoke({"urls": urls})
        
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["url"] == "https://example.com/test"
    
    def test_load_web_documents_empty_urls(self):
        """Test loading documents with empty URL list."""
        result = load_web_documents.invoke({"urls": []})
        
        assert result == []


class TestExtractUrlsFromSearchResults:
    """Test cases for extract_urls_from_search_results tool."""
    
    def test_extract_urls_success(self, mock_search_results):
        """Test successful URL extraction."""
        result = extract_urls_from_search_results.invoke({"search_results": mock_search_results})
        
        assert len(result) == 2
        assert "https://example.com/ai-education" in result
        assert "https://example.com/ai-trends" in result
    
    def test_extract_urls_invalid_results(self):
        """Test URL extraction with invalid data."""
        invalid_results = [
            {"url": "invalid-url", "title": "Test"},
            {"title": "No URL"},
            {"url": "https://valid.com", "title": "Valid"}
        ]
        
        result = extract_urls_from_search_results.invoke({"search_results": invalid_results})
        
        assert len(result) == 1
        assert result[0] == "https://valid.com"


class TestCreateVectorIndex:
    """Test cases for create_vector_index tool."""
    
    @patch('src.tools.VectorStoreIndex.from_documents')
    @patch('src.tools.LlamaDocument')
    def test_create_vector_index_success(self, mock_doc_class, mock_index_class, mock_documents):
        """Test successful vector index creation."""
        mock_index = Mock()
        mock_index.vector_store._data.embedding_dict = {"chunk1": [0.1, 0.2], "chunk2": [0.3, 0.4]}
        mock_index_class.return_value = mock_index
        
        result = create_vector_index.invoke({"documents": mock_documents})
        
        assert "document_count" in result
        assert result["document_count"] == 2
        assert "embedding_model" in result
        assert "_index" in result
    
    def test_create_vector_index_empty_documents(self):
        """Test vector index creation with empty documents."""
        result = create_vector_index.invoke({"documents": []})
        
        assert "error" in result


class TestGenerateOutline:
    """Test cases for generate_outline tool."""
    
    @patch('src.tools.get_openai_llm')
    def test_generate_outline_success(self, mock_llm_func, mock_vector_index):
        """Test successful outline generation."""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = json.dumps({
            "topic": "AI in Education",
            "objective": "Overview of AI in education",
            "slide_titles": ["Title", "Agenda", "Introduction"],
            "target_audience": "Educators",
            "duration_minutes": 15
        })
        mock_llm_func.return_value = mock_llm
        
        mock_query_engine = Mock()
        mock_query_engine.query.return_value.response = "AI education overview content"
        mock_vector_index["_index"].as_query_engine.return_value = mock_query_engine
        
        result = generate_outline.invoke({
            "topic": "AI in Education", 
            "index_metadata": mock_vector_index
        })
        
        assert result["topic"] == "AI in Education"
        assert "slide_titles" in result
        assert len(result["slide_titles"]) == 3
    
    def test_generate_outline_no_index(self):
        """Test outline generation without vector index."""
        result = generate_outline.invoke({
            "topic": "Test Topic",
            "index_metadata": {}
        })
        
        assert "error" in result


class TestGenerateSlideContent:
    """Test cases for generate_slide_content tool."""
    
    @patch('src.tools.get_openai_llm')
    def test_generate_slide_content_success(self, mock_llm_func, mock_vector_index, sample_presentation_outline):
        """Test successful slide content generation."""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = json.dumps({
            "title": "Introduction",
            "bullets": ["Point 1", "Point 2", "Point 3"],
            "speaker_notes": "Detailed notes with citations [1]",
            "slide_type": "content"
        })
        mock_llm_func.return_value = mock_llm
        
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.response = "Generated content"
        mock_response.source_nodes = []
        mock_query_engine.query.return_value = mock_response
        mock_vector_index["_index"].as_query_engine.return_value = mock_query_engine
        
        result = generate_slide_content.invoke({
            "outline": sample_presentation_outline,
            "index_metadata": mock_vector_index
        })
        
        assert len(result) > 0
        assert all("title" in slide for slide in result)
        assert all("bullets" in slide for slide in result)
    
    def test_generate_slide_content_no_index(self, sample_presentation_outline):
        """Test slide content generation without vector index.""" 
        result = generate_slide_content.invoke({
            "outline": sample_presentation_outline,
            "index_metadata": {}
        })
        
        assert len(result) == 1
        assert "error" in result[0]


class TestCreatePresentation:
    """Test cases for create_presentation tool."""
    
    @patch('src.tools.Presentation')
    @patch('src.tools.os.makedirs')
    def test_create_presentation_success(self, mock_makedirs, mock_presentation_class, sample_slide_specs, temp_dir):
        """Test successful presentation creation."""
        mock_prs = Mock()
        mock_slide = Mock()
        mock_prs.slides.add_slide.return_value = mock_slide
        mock_slide.shapes.title.text = ""
        mock_slide.placeholders = {1: Mock()}
        mock_slide.notes_slide.notes_text_frame.text = ""
        mock_prs.slide_layouts = [Mock(), Mock()]
        mock_presentation_class.return_value = mock_prs
        
        with patch('src.tools.settings.default_output_dir', temp_dir):
            result = create_presentation.invoke({
                "slide_specs": sample_slide_specs,
                "template_path": None
            })
        
        assert not result.startswith("Error:")
        assert result.endswith(".pptx")
        mock_prs.save.assert_called_once()
    
    def test_create_presentation_empty_specs(self, temp_dir):
        """Test presentation creation with empty slide specs."""
        with patch('src.tools.settings.default_output_dir', temp_dir):
            result = create_presentation.invoke({
                "slide_specs": [],
                "template_path": None
            })
        
        # Should still create a basic presentation
        assert not result.startswith("Error:")


class TestDeduplicateReferences:
    """Test cases for deduplicate_references tool."""
    
    def test_deduplicate_references_success(self):
        """Test successful reference deduplication."""
        refs = [
            "https://example.com/1",
            "https://example.com/2", 
            "https://example.com/1",  # Duplicate
            "https://example.com/3",
            "https://example.com/2"   # Duplicate
        ]
        
        result = deduplicate_references.invoke({"references": refs})
        
        assert len(result) == 3
        assert result == ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
    
    def test_deduplicate_references_empty(self):
        """Test deduplication with empty list."""
        result = deduplicate_references.invoke({"references": []})
        
        assert result == []
    
    def test_deduplicate_references_with_none(self):
        """Test deduplication with None values."""
        refs = [
            "https://example.com/1",
            None,
            "https://example.com/2",
            "",
            "https://example.com/1"
        ]
        
        result = deduplicate_references.invoke({"references": refs})
        
        assert len(result) == 2
        assert None not in result
        assert "" not in result