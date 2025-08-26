"""Integration tests for the deck builder workflow."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os
from src.deck_builder_agent import build_deck_sync, create_deck_builder_graph


class TestIntegrationWorkflow:
    """Integration tests for the complete workflow."""
    
    @patch('src.tools.requests.get')
    @patch('src.tools.WebBaseLoader')
    @patch('src.tools.VectorStoreIndex.from_documents')
    @patch('src.tools.get_openai_llm')
    @patch('src.tools.Presentation')
    def test_complete_workflow_integration(
        self, 
        mock_presentation,
        mock_llm_func,
        mock_index_class,
        mock_loader_class,
        mock_requests,
        mock_api_keys,
        temp_dir
    ):
        """Test the complete workflow end-to-end with mocked dependencies."""
        
        # Mock Brave Search API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/ai-education",
                        "title": "AI in Education",
                        "description": "Overview of AI in education",
                        "age": "2024-01-15"
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response
        
        # Mock document loading
        mock_loader = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "AI is transforming education through personalized learning."
        mock_doc.metadata = {"title": "AI Education", "source": "https://example.com/ai-education"}
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        # Mock vector index
        mock_index = Mock()
        mock_index.vector_store._data.embedding_dict = {"chunk1": [0.1, 0.2]}
        
        mock_query_engine = Mock()
        mock_query_response = Mock()
        mock_query_response.response = "Generated content based on research"
        mock_query_response.source_nodes = []
        mock_query_engine.query.return_value = mock_query_response
        mock_index.as_query_engine.return_value = mock_query_engine
        
        mock_index_class.return_value = mock_index
        
        # Mock LLM responses
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            # Outline generation response
            Mock(content='{"topic": "AI in Education", "objective": "Overview", "slide_titles": ["Title", "Introduction", "Applications"], "target_audience": "Educators", "duration_minutes": 10}'),
            # Slide content response
            Mock(content='{"title": "Introduction", "bullets": ["AI transforms education", "Personalized learning"], "speaker_notes": "AI overview [1]", "slide_type": "content"}')
        ]
        mock_llm_func.return_value = mock_llm
        
        # Mock PowerPoint creation
        mock_prs = Mock()
        mock_slide = Mock()
        mock_prs.slides.add_slide.return_value = mock_slide
        mock_slide.shapes.title.text = ""
        mock_slide.placeholders = {1: Mock()}
        mock_slide.notes_slide.notes_text_frame.text = ""
        mock_prs.slide_layouts = [Mock(), Mock()]
        mock_presentation.return_value = mock_prs
        
        # Patch settings for output directory
        with patch('src.tools.settings.default_output_dir', temp_dir):
            # Run the complete workflow
            result = build_deck_sync("AI in Education trends and applications")
        
        # Verify successful completion
        assert result["success"] is True
        assert result["slide_count"] >= 1
        assert result["output_path"].endswith(".pptx")
        assert "completed" in result["status"].lower()
        
        # Verify API calls were made
        mock_requests.assert_called_once()
        mock_loader_class.assert_called_once()
        mock_index_class.assert_called_once()
        mock_llm.invoke.assert_called()
        mock_prs.save.assert_called_once()
    
    @patch('src.tools.requests.get')
    def test_workflow_with_api_failure(self, mock_requests, mock_api_keys):
        """Test workflow behavior when API calls fail."""
        # Mock API failure
        mock_requests.side_effect = Exception("API unavailable")
        
        result = build_deck_sync("Test topic")
        
        # Should handle gracefully and report failure
        assert result["success"] is False
        assert "error_message" in result or "failed" in result["status"].lower()
    
    def test_graph_compilation(self):
        """Test that the LangGraph compiles without errors."""
        try:
            graph = create_deck_builder_graph()
            assert graph is not None
            
            # Test that we can get node information
            nodes = list(graph.nodes.keys())
            assert len(nodes) > 0
            
        except Exception as e:
            pytest.fail(f"Graph compilation failed: {e}")
    
    @patch('src.deck_builder_agent.deck_builder_graph')
    def test_error_handling_in_workflow(self, mock_graph):
        """Test error handling throughout the workflow."""
        # Simulate workflow failure
        mock_graph.invoke.side_effect = Exception("Workflow error")
        
        result = build_deck_sync("Test topic")
        
        assert result["success"] is False
        assert "error_message" in result
        assert result["error_message"] == "Workflow error"
    
    def test_workflow_with_template_path(self, temp_dir):
        """Test workflow with custom template path."""
        template_path = os.path.join(temp_dir, "template.pptx")
        
        # Create a mock template file
        with open(template_path, 'wb') as f:
            f.write(b"mock pptx data")
        
        with patch('src.deck_builder_agent.deck_builder_graph') as mock_graph:
            mock_final_state = {
                "output_path": "test_deck.pptx",
                "slide_specs": [],
                "references": [],
                "status": "completed",
                "messages": []
            }
            mock_graph.invoke.return_value = mock_final_state
            
            result = build_deck_sync(
                "Test topic",
                template_path=template_path
            )
            
            assert result["success"] is True
            # Verify template path was passed in the initial state
            call_args = mock_graph.invoke.call_args
            initial_state = call_args[0][0]
            assert initial_state["template_path"] == template_path


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    @patch('src.tools.search_web')
    def test_search_failure_recovery(self, mock_search, mock_api_keys):
        """Test recovery when search fails."""
        mock_search.invoke.return_value = []
        
        with patch('src.deck_builder_agent.deck_builder_graph') as mock_graph:
            mock_graph.invoke.side_effect = Exception("Search failed")
            
            result = build_deck_sync("Test topic")
            
            assert result["success"] is False
    
    @patch('src.tools.create_vector_index')
    def test_indexing_failure_recovery(self, mock_index):
        """Test recovery when vector indexing fails."""
        mock_index.invoke.return_value = {"error": "Indexing failed"}
        
        with patch('src.deck_builder_agent.deck_builder_graph') as mock_graph:
            mock_graph.invoke.side_effect = Exception("Indexing failed")
            
            result = build_deck_sync("Test topic")
            
            assert result["success"] is False