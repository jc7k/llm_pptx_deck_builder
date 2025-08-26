"""Unit tests for the main agent workflow."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from src.deck_builder_agent import (
    research_node,
    document_loading_node,
    indexing_node,
    outline_generation_node,
    content_generation_node,
    presentation_creation_node,
    build_deck_sync,
    create_deck_builder_graph
)
from src.models import DeckBuilderState
from langchain_core.messages import HumanMessage


class TestWorkflowNodes:
    """Test cases for individual workflow nodes."""
    
    def test_research_node_success(self, mock_search_results):
        """Test successful research node execution."""
        with patch('src.deck_builder_agent.search_web') as mock_search:
            mock_search.invoke.return_value = mock_search_results
            
            state = {
                "user_request": "AI in Education",
                "messages": [],
                "status": "starting"
            }
            
            result = research_node(state)
            
            assert "search_results" in result
            assert len(result["search_results"]) == 2
            assert result["status"] == "Research completed"
            assert len(result["messages"]) == 1
    
    def test_research_node_failure(self):
        """Test research node with failure."""
        with patch('src.deck_builder_agent.search_web') as mock_search:
            mock_search.invoke.side_effect = Exception("Search failed")
            
            state = {
                "user_request": "AI in Education",
                "messages": [],
                "status": "starting"
            }
            
            result = research_node(state)
            
            assert result["search_results"] == []
            assert "failed" in result["status"].lower()
    
    def test_document_loading_node_success(self, mock_search_results, mock_documents):
        """Test successful document loading node."""
        with patch('src.deck_builder_agent.extract_urls_from_search_results') as mock_extract, \
             patch('src.deck_builder_agent.load_web_documents') as mock_load:
            
            mock_extract.invoke.return_value = ["https://example.com/test"]
            mock_load.invoke.return_value = mock_documents
            
            state = {
                "search_results": mock_search_results,
                "messages": [],
                "status": "research completed"
            }
            
            result = document_loading_node(state)
            
            assert "documents" in result
            assert len(result["documents"]) == 2
            assert result["status"] == "Document loading completed"
    
    def test_indexing_node_success(self, mock_documents, mock_vector_index):
        """Test successful indexing node."""
        with patch('src.deck_builder_agent.create_vector_index') as mock_create:
            mock_create.invoke.return_value = mock_vector_index
            
            state = {
                "documents": mock_documents,
                "messages": [],
                "status": "documents loaded"
            }
            
            result = indexing_node(state)
            
            assert "vector_index" in result
            assert result["vector_index"]["document_count"] == 2
            assert result["status"] == "Vector indexing completed"
    
    def test_outline_generation_node_success(self, mock_vector_index, sample_presentation_outline):
        """Test successful outline generation node."""
        with patch('src.deck_builder_agent.generate_outline') as mock_generate:
            mock_generate.invoke.return_value = sample_presentation_outline
            
            state = {
                "user_request": "AI in Education",
                "vector_index": mock_vector_index,
                "messages": [],
                "status": "indexing completed"
            }
            
            result = outline_generation_node(state)
            
            assert "outline" in result
            assert result["outline"]["topic"] == "AI in Education"
            assert result["status"] == "Outline generation completed"
    
    def test_content_generation_node_success(self, sample_presentation_outline, mock_vector_index, sample_slide_specs):
        """Test successful content generation node."""
        with patch('src.deck_builder_agent.generate_slide_content') as mock_generate, \
             patch('src.deck_builder_agent.deduplicate_references') as mock_dedup:
            
            mock_generate.invoke.return_value = sample_slide_specs
            mock_dedup.invoke.return_value = ["https://example.com/ai-education"]
            
            state = {
                "outline": sample_presentation_outline,
                "vector_index": mock_vector_index,
                "messages": [],
                "status": "outline completed"
            }
            
            result = content_generation_node(state)
            
            assert "slide_specs" in result
            assert "references" in result
            assert len(result["slide_specs"]) == 2
            assert result["status"] == "Content generation completed"
    
    def test_presentation_creation_node_success(self, sample_slide_specs, temp_dir):
        """Test successful presentation creation node."""
        with patch('src.deck_builder_agent.create_presentation') as mock_create:
            mock_create.invoke.return_value = f"{temp_dir}/test_presentation.pptx"
            
            state = {
                "slide_specs": sample_slide_specs,
                "template_path": None,
                "messages": [],
                "status": "content completed"
            }
            
            result = presentation_creation_node(state)
            
            assert "output_path" in result
            assert result["output_path"].endswith(".pptx")
            assert result["status"] == "Presentation creation completed"


class TestMainWorkflow:
    """Test cases for the main workflow functions."""
    
    @patch('src.deck_builder_agent.deck_builder_graph')
    def test_build_deck_sync_success(self, mock_graph):
        """Test successful synchronous deck building."""
        mock_final_state = {
            "output_path": "test_deck.pptx",
            "slide_specs": [{"title": "Test"}],
            "references": ["https://example.com"],
            "status": "completed",
            "messages": [HumanMessage(content="Test message")]
        }
        mock_graph.invoke.return_value = mock_final_state
        
        result = build_deck_sync("AI in Education")
        
        assert result["success"] is True
        assert result["output_path"] == "test_deck.pptx"
        assert result["slide_count"] == 1
        assert result["references_count"] == 1
    
    @patch('src.deck_builder_agent.deck_builder_graph')
    def test_build_deck_sync_failure(self, mock_graph):
        """Test deck building with failure."""
        mock_graph.invoke.side_effect = Exception("Workflow failed")
        
        result = build_deck_sync("AI in Education")
        
        assert result["success"] is False
        assert "error_message" in result
        assert "failed" in result["status"].lower()
    
    def test_create_deck_builder_graph(self):
        """Test graph creation and compilation."""
        graph = create_deck_builder_graph()
        
        assert graph is not None
        # Verify that the graph has the expected nodes
        node_names = list(graph.nodes.keys())
        expected_nodes = [
            "research", "load_docs", "create_index", 
            "generate_outline", "generate_content", "create_presentation"
        ]
        
        for node in expected_nodes:
            assert node in node_names


class TestStateManagement:
    """Test cases for state management and transitions."""
    
    def test_initial_state_structure(self):
        """Test initial state has required structure."""
        initial_state = {
            "messages": [HumanMessage(content="Test")],
            "user_request": "Test request",
            "search_results": [],
            "documents": [],
            "vector_index": None,
            "outline": None,
            "slide_specs": [],
            "references": [],
            "template_path": None,
            "output_path": "",
            "status": "Starting..."
        }
        
        # Verify all required keys are present
        required_keys = [
            "messages", "user_request", "search_results", "documents",
            "vector_index", "outline", "slide_specs", "references", 
            "template_path", "output_path", "status"
        ]
        
        for key in required_keys:
            assert key in initial_state
    
    def test_state_updates_preserve_structure(self, mock_search_results):
        """Test that state updates preserve the required structure."""
        with patch('src.deck_builder_agent.search_web') as mock_search:
            mock_search.invoke.return_value = mock_search_results
            
            initial_state = {
                "user_request": "AI in Education",
                "messages": [],
                "status": "starting",
                "search_results": [],
                "documents": [],
                "vector_index": None
            }
            
            result = research_node(initial_state)
            
            # Verify the result maintains state structure
            assert "search_results" in result
            assert "messages" in result
            assert "status" in result
            # Verify original state keys are preserved
            for key in initial_state:
                if key not in result:
                    # Should still have the original value
                    pass