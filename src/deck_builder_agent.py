"""Main LangGraph workflow for the LLM PPTX Deck Builder."""

import time
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .models import DeckBuilderState

# Phoenix observability imports
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import AttributeValue
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
from .tools import (
    search_web,
    load_web_documents,
    extract_urls_from_search_results,
    create_vector_index,
    generate_outline,
    generate_slide_content,
    create_presentation,
    deduplicate_references,
)

# Global progress callback for verbose mode
_progress_callback: Optional[Callable[[str, str], None]] = None

# Phoenix tracer
_tracer = trace.get_tracer(__name__) if PHOENIX_AVAILABLE else None


def set_progress_callback(callback: Optional[Callable[[str, str], None]]):
    """Set global progress callback for verbose mode."""
    global _progress_callback
    _progress_callback = callback


def _log_progress(step: str, message: str):
    """Log progress to callback if available."""
    if _progress_callback:
        _progress_callback(step, message)
    else:
        # Fallback to simple print
        print(f"[{time.strftime('%H:%M:%S')}] {step}: {message}", flush=True)


def _create_span_attributes(
    node_name: str,
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, AttributeValue]:
    """Create standardized span attributes for workflow nodes."""
    attributes = {
        "workflow.node.name": node_name,
        "workflow.node.type": "langgraph_node",
        "input.size": len(str(input_data)) if input_data else 0,
    }
    
    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"metadata.{key}"] = value
    
    # Node-specific input attributes
    if node_name == "research":
        topic = input_data.get("topic", "")
        attributes.update({
            "research.topic": topic[:100] if topic else "",  # Truncate long topics
            "research.topic_length": len(topic) if topic else 0,
        })
        
        # Output attributes for research
        if output_data:
            research_results = output_data.get("research_results", {})
            attributes.update({
                "research.urls_found": len(research_results.get("urls", [])),
                "research.search_queries": len(research_results.get("search_queries", [])),
                "research.total_results": research_results.get("total_results", 0),
                "research.success": research_results.get("success", False),
            })
            
            # Add error tracking
            if not research_results.get("success"):
                attributes["research.has_errors"] = True
                error_msg = research_results.get("error", "")
                if error_msg:
                    attributes["research.error_type"] = error_msg[:50]  # Truncate error
    
    elif node_name == "document_loading":
        urls = input_data.get("research_results", {}).get("urls", [])
        attributes.update({
            "loading.input_urls": len(urls),
            "loading.urls_total": len(urls),
        })
        
        # Output attributes for document loading
        if output_data:
            loading_results = output_data.get("loading_results", {})
            documents = output_data.get("documents", [])
            attributes.update({
                "loading.documents_loaded": len(documents),
                "loading.successful_loads": loading_results.get("successful_loads", 0),
                "loading.failed_loads": loading_results.get("failed_loads", 0),
                "loading.total_content_length": loading_results.get("total_content_length", 0),
                "loading.success_rate": loading_results.get("successful_loads", 0) / max(len(urls), 1),
            })
            
            # Add error tracking
            if loading_results.get("failed_loads", 0) > 0:
                attributes["loading.has_failures"] = True
                attributes["loading.failure_rate"] = loading_results.get("failed_loads", 0) / max(len(urls), 1)
    
    elif node_name == "indexing":
        documents = input_data.get("documents", [])
        attributes.update({
            "indexing.input_documents": len(documents),
            "indexing.total_documents": len(documents),
        })
        
        # Output attributes for indexing
        if output_data:
            indexing_results = output_data.get("indexing_results", {})
            attributes.update({
                "indexing.indexed_chunks": indexing_results.get("indexed_chunks", 0),
                "indexing.index_size": indexing_results.get("index_size", 0),
                "indexing.success": output_data.get("index_id") is not None,
            })
            
            # Add error tracking
            if not output_data.get("index_id"):
                attributes["indexing.has_errors"] = True
                attributes["indexing.chunks_indexed"] = 0
            else:
                attributes["indexing.chunks_per_document"] = indexing_results.get("indexed_chunks", 0) / max(len(documents), 1)
    
    elif node_name == "outline_generation":
        topic = input_data.get("topic", "")
        index_available = bool(input_data.get("index_id"))
        attributes.update({
            "outline.topic": topic[:100] if topic else "",
            "outline.topic_length": len(topic) if topic else 0,
            "outline.has_rag_context": index_available,
        })
        
        # Output attributes for outline generation
        if output_data:
            outline = output_data.get("outline")
            outline_results = output_data.get("outline_results", {})
            attributes.update({
                "outline.slides_planned": outline_results.get("slide_count", 0),
                "outline.generation_successful": outline_results.get("generation_successful", False),
                "outline.validation_passed": outline_results.get("validation_passed", False),
            })
            
            # Add error tracking
            if not outline_results.get("generation_successful"):
                attributes["outline.has_errors"] = True
            
            # Add slide type distribution if available
            if outline and outline.get("slide_specs"):
                slide_specs = outline.get("slide_specs", [])
                attributes["outline.content_slides"] = len([s for s in slide_specs if s.get("title", "").lower() not in ["agenda", "references", "conclusion"]])
    
    elif node_name == "content_generation":
        outline = input_data.get("outline")
        index_available = bool(input_data.get("index_id"))
        slide_count = len(outline.get("slide_specs", [])) if outline else 0
        
        attributes.update({
            "content.slides_to_generate": slide_count,
            "content.has_rag_context": index_available,
        })
        
        # Output attributes for content generation
        if output_data:
            content_results = output_data.get("content_results", {})
            detailed_content = output_data.get("detailed_content")
            attributes.update({
                "content.slides_generated": content_results.get("slides_generated", 0),
                "content.total_references": content_results.get("total_references", 0),
                "content.generation_successful": content_results.get("generation_successful", False),
                "content.validation_passed": content_results.get("validation_passed", False),
            })
            
            # Add error tracking
            if not content_results.get("generation_successful"):
                attributes["content.has_errors"] = True
            
            # Calculate content metrics
            if detailed_content and detailed_content.get("slide_specs"):
                slide_specs = detailed_content.get("slide_specs", [])
                total_bullets = sum(len(slide.get("bullets", [])) for slide in slide_specs)
                total_notes_length = sum(len(slide.get("notes", "")) for slide in slide_specs)
                
                attributes.update({
                    "content.total_bullet_points": total_bullets,
                    "content.total_notes_length": total_notes_length,
                    "content.avg_bullets_per_slide": total_bullets / max(len(slide_specs), 1),
                    "content.avg_notes_per_slide": total_notes_length / max(len(slide_specs), 1),
                })
    
    elif node_name == "presentation_creation":
        detailed_content = input_data.get("detailed_content")
        template_used = bool(input_data.get("template_path"))
        slide_count = len(detailed_content.get("slide_specs", [])) if detailed_content else 0
        
        attributes.update({
            "presentation.slides_to_create": slide_count,
            "presentation.template_used": template_used,
            "presentation.output_path": input_data.get("output_path", ""),
        })
        
        # Output attributes for presentation creation
        if output_data:
            presentation_results = output_data.get("presentation_results", {})
            attributes.update({
                "presentation.slides_created": presentation_results.get("slides_created", 0),
                "presentation.file_size_bytes": presentation_results.get("file_size", 0),
                "presentation.creation_successful": presentation_results.get("creation_successful", False),
            })
            
            # Add error tracking
            if not presentation_results.get("creation_successful"):
                attributes["presentation.has_errors"] = True
            
            # Calculate file size in KB/MB for easier reading
            file_size = presentation_results.get("file_size", 0)
            if file_size > 0:
                attributes["presentation.file_size_kb"] = round(file_size / 1024, 2)
                if file_size > 1024 * 1024:
                    attributes["presentation.file_size_mb"] = round(file_size / (1024 * 1024), 2)
    
    # Add output size if available
    if output_data:
        attributes["output.size"] = len(str(output_data))
        
        # Add general success indicators
        success_indicators = [
            "success", "generation_successful", "creation_successful", 
            "validation_passed", "loading_successful"
        ]
        
        for indicator in success_indicators:
            if indicator in output_data:
                attributes[f"output.{indicator}"] = bool(output_data[indicator])
    
    return attributes


def _set_span_status(span, success: bool, error_message: Optional[str] = None):
    """Set span status based on operation success/failure."""
    if not PHOENIX_AVAILABLE or not span:
        return
    
    if success:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, error_message or "Operation failed"))
        if error_message:
            span.set_attribute("error.message", error_message)


def _set_span_status(span, success: bool, error_message: Optional[str] = None):
    """Set span status and error information."""
    try:
        from opentelemetry.trace import Status, StatusCode
        
        if success:
            span.set_status(Status(StatusCode.OK))
            span.set_attribute("execution.success", True)
        else:
            span.set_status(Status(StatusCode.ERROR, error_message or "Unknown error"))
            span.set_attribute("execution.success", False)
            span.set_attribute("execution.error", True)
            
            if error_message:
                span.set_attribute("execution.error_message", error_message)
                # Record the exception as an event
                span.record_exception(Exception(error_message))
                
    except Exception as e:
        # Fallback - don't let tracing errors break the application
        print(f"Warning: Failed to set span status: {e}")

def _determine_node_success(node_name: str, result: Dict[str, Any]) -> bool:
    """Determine if a node execution was successful based on result content."""
    if not result:
        return False
    
    # Node-specific success criteria
    if node_name == "research":
        research_results = result.get("research_results", {})
        return research_results.get("success", False) and len(research_results.get("urls", [])) > 0
    
    elif node_name == "document_loading":
        loading_results = result.get("loading_results", {})
        return loading_results.get("successful_loads", 0) > 0
    
    elif node_name == "indexing":
        return result.get("index_id") is not None
    
    elif node_name == "outline_generation":
        outline_results = result.get("outline_results", {})
        return (outline_results.get("generation_successful", False) and 
                outline_results.get("slide_count", 0) > 0)
    
    elif node_name == "content_generation":
        content_results = result.get("content_results", {})
        return (content_results.get("generation_successful", False) and 
                content_results.get("slides_generated", 0) > 0)
    
    elif node_name == "presentation_creation":
        presentation_results = result.get("presentation_results", {})
        return presentation_results.get("creation_successful", False)
    
    # Default: check for general success indicators
    return result.get("success", True)

def _extract_error_message(node_name: str, result: Dict[str, Any]) -> str:
    """Extract error message from node result for tracing."""
    if not result:
        return f"No result returned from {node_name}"
    
    # Look for common error fields
    error_fields = ["error", "error_message", "failure_reason"]
    for field in error_fields:
        if field in result and result[field]:
            return str(result[field])[:200]  # Truncate long error messages
    
    # Node-specific error extraction
    if node_name == "research":
        research_results = result.get("research_results", {})
        if not research_results.get("success"):
            return research_results.get("error", "Research failed with unknown error")
    
    elif node_name == "document_loading":
        loading_results = result.get("loading_results", {})
        failed_loads = loading_results.get("failed_loads", 0)
        total_urls = loading_results.get("total_urls", 0)
        if failed_loads > 0:
            return f"Failed to load {failed_loads}/{total_urls} documents"
    
    elif node_name == "indexing":
        if not result.get("index_id"):
            return "Failed to create vector index"
    
    elif node_name == "outline_generation":
        outline_results = result.get("outline_results", {})
        if not outline_results.get("generation_successful"):
            return "LLM failed to generate valid outline"
    
    elif node_name == "content_generation":
        content_results = result.get("content_results", {})
        if not content_results.get("generation_successful"):
            return "LLM failed to generate slide content"
    
    elif node_name == "presentation_creation":
        presentation_results = result.get("presentation_results", {})
        if not presentation_results.get("creation_successful"):
            return "Failed to create PowerPoint file"
    
    return f"Unknown error in {node_name}"

def _trace_workflow_node(node_name: str):
    """Decorator to add Phoenix tracing to workflow nodes."""
    def decorator(func):
        def wrapper(state: DeckBuilderState) -> Dict[str, Any]:
            if not PHOENIX_AVAILABLE or not _tracer:
                return func(state)
            
            with _tracer.start_as_current_span(f"workflow.{node_name}") as span:
                start_time = time.time()
                
                try:
                    # Set initial attributes
                    initial_attributes = _create_span_attributes(node_name, state)
                    for key, value in initial_attributes.items():
                        span.set_attribute(key, value)
                    
                    # Add span event for node start
                    span.add_event(f"{node_name}_node_started", {
                        "timestamp": start_time,
                        "node_name": node_name
                    })
                    
                    # Execute the node function
                    result = func(state)
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    span.set_attribute("execution.duration_seconds", execution_time)
                    
                    # Set output attributes
                    output_attributes = _create_span_attributes(node_name, state, result)
                    for key, value in output_attributes.items():
                        if key not in initial_attributes:  # Avoid duplicates
                            span.set_attribute(key, value)
                    
                    # Determine success based on result content
                    success = _determine_node_success(node_name, result)
                    
                    # Set success/error status
                    if success:
                        _set_span_status(span, True)
                        span.add_event(f"{node_name}_node_completed", {
                            "execution_time": execution_time,
                            "success": True
                        })
                    else:
                        error_msg = _extract_error_message(node_name, result)
                        _set_span_status(span, False, error_msg)
                        span.add_event(f"{node_name}_node_failed", {
                            "execution_time": execution_time,
                            "success": False,
                            "error_message": error_msg
                        })
                    
                    return result
                    
                except Exception as e:
                    # Handle unexpected exceptions
                    execution_time = time.time() - start_time
                    span.set_attribute("execution.duration_seconds", execution_time)
                    span.set_attribute("execution.exception_type", type(e).__name__)
                    
                    error_msg = f"Unexpected error in {node_name}: {str(e)}"
                    _set_span_status(span, False, error_msg)
                    
                    # Add exception event
                    span.add_event(f"{node_name}_node_exception", {
                        "execution_time": execution_time,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    })
                    
                    # Re-raise the exception
                    raise
        
        return wrapper
    return decorator


@_trace_workflow_node("research")
def research_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Research node: Search web using Brave Search API."""
    try:
        user_request = state["user_request"]

        _log_progress("🔍 RESEARCH", f"Starting web search for: {user_request}")
        _log_progress("🔍 RESEARCH", "Applying rate limiting for Brave Search API...")

        # Update status
        state["status"] = "Researching topic..."

        # Create enhanced search query for better results
        enhanced_query = f"{user_request} statistics data trends 2025 report analysis"

        # Search for relevant information
        _log_progress(
            "🔍 RESEARCH", f"Enhanced search query: {enhanced_query[:100]}..."
        )
        _log_progress("🔍 RESEARCH", "Querying Brave Search API...")
        search_results = search_web.invoke({"query": enhanced_query, "count": 15})

        _log_progress("🔍 RESEARCH", f"✅ Found {len(search_results)} search results")
        
        # Debug: Log the structure of search results
        if search_results:
            _log_progress("🔍 RESEARCH", f"First result structure: {search_results[0]}")

        # Extract URLs for document loading
        urls = []
        for result in search_results[:5]:
            if isinstance(result, dict) and result.get("url"):
                urls.append(result["url"])
            elif hasattr(result, 'url') and result.url:
                urls.append(result.url)
                
        _log_progress("🔍 RESEARCH", f"Extracted URLs: {urls}")
        
        if urls:
            _log_progress("🔍 RESEARCH", f"Top sources: {', '.join(urls)}")

        # Add a human message about the research
        research_message = HumanMessage(
            content=f"Found {len(search_results)} search results for: {user_request}"
        )

        # Create research_results structure for next nodes
        research_results = {
            "success": len(search_results) > 0,
            "urls": urls,
            "search_queries": [enhanced_query],
            "total_results": len(search_results),
            "error": None if len(search_results) > 0 else "No search results found"
        }

        _log_progress("🔍 RESEARCH", f"Research results structure: {research_results}")

        return {
            "search_results": search_results,
            "research_results": research_results,
            "topic": user_request,  # Pass topic to next nodes
            "messages": state["messages"] + [research_message],
            "status": "Research completed",
        }

    except Exception as e:
        _log_progress("🔍 RESEARCH", f"❌ Research failed: {str(e)}")
        error_message = HumanMessage(content=f"Research failed: {str(e)}")
        
        # Create failed research_results structure
        research_results = {
            "success": False,
            "urls": [],
            "search_queries": [],
            "total_results": 0,
            "error": str(e)
        }
        
        return {
            "search_results": [],
            "research_results": research_results,
            "topic": state.get("user_request", ""),
            "messages": state["messages"] + [error_message],
            "status": f"Research failed: {str(e)}",
        }


@_trace_workflow_node("document_loading")
def document_loading_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Load documents from URLs using LangChain WebBaseLoader."""
    _log_progress("📄 LOADING", "Starting document loading phase")
    
    # Debug: Log the entire state to see what we're receiving
    _log_progress("📄 LOADING", f"Received state keys: {list(state.keys())}")
    
    research_results = state.get("research_results", {})
    _log_progress("📄 LOADING", f"Research results: {research_results}")
    
    urls = research_results.get("urls", [])
    _log_progress("📄 LOADING", f"URLs from research_results: {urls}")
    
    if not urls:
        # Try alternative data sources
        search_results = state.get("search_results", [])
        if search_results:
            _log_progress("📄 LOADING", "No URLs in research_results, trying search_results")
            urls = []
            for result in search_results[:5]:
                if isinstance(result, dict) and result.get("url"):
                    urls.append(result["url"])
            _log_progress("📄 LOADING", f"Extracted URLs from search_results: {urls}")
    
    if not urls:
        _log_progress("📄 LOADING", "No URLs provided for document loading")
        return {
            "documents": [],
            "loading_results": {
                "total_urls": 0,
                "successful_loads": 0,
                "failed_loads": 0,
                "total_content_length": 0
            }
        }
    
    _log_progress("📄 LOADING", f"Loading {len(urls)} documents")
    
    documents = []
    successful_loads = 0
    failed_loads = 0
    total_content_length = 0
    
    # Use the load_web_documents tool which handles rate limiting and retries
    try:
        _log_progress("📄 LOADING", f"Calling load_web_documents with URLs: {urls}")
        result = load_web_documents.invoke({"urls": urls})
        _log_progress("📄 LOADING", f"load_web_documents result type: {type(result)}")
        
        if result and isinstance(result, list):
            documents = result
            successful_loads = len(documents)
            total_content_length = sum(len(doc.page_content) for doc in documents if hasattr(doc, 'page_content'))
            failed_loads = len(urls) - successful_loads
            _log_progress("📄 LOADING", f"Successfully loaded {successful_loads} documents")
        else:
            _log_progress("📄 LOADING", f"Document loading failed: No documents returned, result={result}")
            failed_loads = len(urls)
    except Exception as e:
        _log_progress("📄 LOADING", f"Error during document loading: {e}")
        failed_loads = len(urls)
    
    loading_results = {
        "total_urls": len(urls),
        "successful_loads": successful_loads,
        "failed_loads": failed_loads,
        "total_content_length": total_content_length
    }
    
    _log_progress("📄 LOADING", f"Document loading complete: {successful_loads} successful, {failed_loads} failed")
    
    return {
        "documents": documents,
        "loading_results": loading_results
    }


@_trace_workflow_node("indexing")
def indexing_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Create vector index from loaded documents."""
    _log_progress("🔍 INDEXING", "Starting indexing phase")
    
    documents = state.get("documents", [])
    if not documents:
        _log_progress("🔍 INDEXING", "No documents provided for indexing")
        return {
            "vector_index": None,  # Use correct state field name
        }
    
    _log_progress("🔍 INDEXING", f"Creating index from {len(documents)} documents")
    
    try:
        # Use the create_vector_index tool
        result = create_vector_index.invoke({"documents": documents})
        _log_progress("🔍 INDEXING", f"create_vector_index result: {result}")
        
        # Check for error first (function returns {"error": str} on failure)
        if "error" in result:
            error_msg = result["error"]
            _log_progress("🔍 INDEXING", f"Indexing failed: {error_msg}")
            return {
                "vector_index": None,
            }
        
        # Success case - result is metadata dict with index_id, chunk_count etc.
        if result and "index_id" in result:
            index_id = result.get("index_id")
            
            # Store complete metadata in vector_index field (matches DeckBuilderState)
            vector_index_metadata = {
                "index_id": index_id,
                "document_count": len(documents),
                "chunk_count": result.get("chunk_count", 0),
                "embedding_model": result.get("embedding_model", ""),
                "created_at": result.get("created_at", ""),
                "success": True
            }
            
            _log_progress("🔍 INDEXING", f"Index created successfully: {index_id}")
            return {
                "vector_index": vector_index_metadata,
            }
        else:
            _log_progress("🔍 INDEXING", f"Indexing failed: Invalid result structure: {result}")
            return {
                "vector_index": None,
            }
    
    except Exception as e:
        _log_progress("🔍 INDEXING", f"Error during indexing: {e}")
        return {
            "vector_index": None,
        }


@_trace_workflow_node("outline_generation")
def outline_generation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Generate presentation outline using LLM with RAG context."""
    _log_progress("📋 OUTLINE", "Starting outline generation phase")
    
    # Use user_request instead of topic (which doesn't exist in state)
    topic = state.get("user_request", "")
    vector_index = state.get("vector_index")  # Get vector_index dict
    
    _log_progress("📋 OUTLINE", f"Topic from user_request: {topic}")
    _log_progress("📋 OUTLINE", f"Vector index data: {vector_index}")
    
    if not topic:
        _log_progress("📋 OUTLINE", "No topic provided for outline generation")
        return {
            "outline": None,
            "outline_results": {
                "generation_successful": False,
                "slide_count": 0,
                "validation_passed": False
            }
        }
    
    if not vector_index:
        _log_progress("📋 OUTLINE", "No vector index provided - generating outline without RAG context")
    
    _log_progress("📋 OUTLINE", f"Generating outline for topic: {topic}")
    
    try:
        # Use the generate_outline tool - pass the full vector_index as index_metadata
        result = generate_outline.invoke({
            "topic": topic,
            "index_metadata": vector_index or {}
        })
        
        if result and not result.get("error"):
            outline = result
            slide_count = len(result.get("slide_titles", [])) if result else 0
            
            outline_results = {
                "generation_successful": True,
                "slide_count": slide_count,
                "validation_passed": True
            }
            
            _log_progress("📋 OUTLINE", f"Outline generated successfully with {slide_count} slides")
            return {
                "outline": outline,
                "outline_results": outline_results
            }
        else:
            error_msg = result.get("error", "Unknown outline generation error") if result else "No result returned"
            _log_progress("📋 OUTLINE", f"Outline generation failed: {error_msg}")
            return {
                "outline": None,
                "outline_results": {
                    "generation_successful": False,
                    "slide_count": 0,
                    "validation_passed": False
                }
            }
    
    except Exception as e:
        _log_progress("📋 OUTLINE", f"Error during outline generation: {e}")
        return {
            "outline": None,
            "outline_results": {
                "generation_successful": False,
                "slide_count": 0,
                "validation_passed": False
            }
        }


@_trace_workflow_node("content_generation")
def content_generation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Generate detailed slide content using LLM with RAG context."""
    _log_progress("📝 CONTENT", "Starting content generation phase")
    
    # Use user_request instead of topic (which doesn't exist in state)
    topic = state.get("user_request", "")
    outline = state.get("outline")
    vector_index = state.get("vector_index")  # Get vector_index dict
    
    _log_progress("📝 CONTENT", f"Topic from user_request: {topic}")
    _log_progress("📝 CONTENT", f"Vector index data: {vector_index}")
    
    if not topic or not outline:
        _log_progress("📝 CONTENT", f"Missing topic ({bool(topic)}) or outline ({bool(outline)}) for content generation")
        return {
            "slide_specs": [],
            "references": [],
            "content_results": {
                "generation_successful": False,
                "slides_generated": 0,
                "validation_passed": False,
                "total_references": 0
            }
        }
    
    if not vector_index:
        _log_progress("📝 CONTENT", "No vector index provided - generating content without RAG context")
    
    slide_count = len(outline.get("slide_titles", [])) if outline else 0
    _log_progress("📝 CONTENT", f"Generating detailed content for {slide_count} slides")
    
    try:
        # Use the generate_slide_content tool - pass outline and index_metadata
        result = generate_slide_content.invoke({
            "outline": outline,
            "index_metadata": vector_index or {}
        })
        
        if result and isinstance(result, list) and len(result) > 0:
            # result is a list of slide specs
            slides_generated = len(result)
            
            # Extract references from slide specs
            all_references = []
            for slide in result:
                slide_refs = slide.get("references", [])
                all_references.extend(slide_refs)
            
            # Deduplicate references
            unique_references = list(dict.fromkeys(all_references))
            
            content_results = {
                "generation_successful": True,
                "slides_generated": slides_generated,
                "validation_passed": True,
                "total_references": len(unique_references)
            }
            
            _log_progress("📝 CONTENT", f"Content generated successfully: {slides_generated} slides, {len(unique_references)} references")
            return {
                "slide_specs": result,
                "references": unique_references,
                "content_results": content_results
            }
        else:
            _log_progress("📝 CONTENT", f"Content generation failed: Invalid or empty result")
            return {
                "slide_specs": [],
                "references": [],
                "content_results": {
                    "generation_successful": False,
                    "slides_generated": 0,
                    "validation_passed": False,
                    "total_references": 0
                }
            }
    
    except Exception as e:
        _log_progress("📝 CONTENT", f"Error during content generation: {e}")
        return {
            "slide_specs": [],
            "references": [],
            "content_results": {
                "generation_successful": False,
                "slides_generated": 0,
                "validation_passed": False,
                "total_references": 0
            }
        }


@_trace_workflow_node("presentation_creation")
def presentation_creation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Create final PowerPoint presentation."""
    _log_progress("📊 PRESENTATION", "Starting presentation creation phase")
    
    detailed_content = state.get("detailed_content")
    template_path = state.get("template_path")
    output_path = state.get("output_path", "presentation.pptx")
    
    if not detailed_content:
        _log_progress("📊 PRESENTATION", "No detailed content provided for presentation creation")
        return {
            "final_output_path": None,
            "presentation_results": {
                "creation_successful": False,
                "slides_created": 0,
                "file_size": 0,
                "template_used": bool(template_path)
            }
        }
    
    _log_progress("📊 PRESENTATION", f"Creating presentation: {output_path}")
    if template_path:
        _log_progress("📊 PRESENTATION", f"Using template: {template_path}")
    
    try:
        # Use the create_presentation tool
        result = create_presentation.invoke({
            "content": detailed_content,
            "output_path": output_path,
            "template_path": template_path
        })
        
        if result and result.get("success"):
            file_size = 0
            slides_created = len(detailed_content.get("slide_specs", [])) if detailed_content else 0
            
            # Try to get file size
            try:
                import os
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
            except Exception:
                pass
            
            presentation_results = {
                "creation_successful": True,
                "slides_created": slides_created,
                "file_size": file_size,
                "template_used": bool(template_path)
            }
            
            _log_progress("📊 PRESENTATION", f"Presentation created successfully: {output_path} ({file_size} bytes)")
            return {
                "final_output_path": output_path,
                "presentation_results": presentation_results,
                "output_path": output_path,  # For backward compatibility
                "slide_specs": detailed_content.get("slide_specs", []),  # For final state
                "references": detailed_content.get("references", [])  # For final state
            }
        else:
            error_msg = result.get("error", "Unknown presentation creation error") if result else "No result returned"
            _log_progress("📊 PRESENTATION", f"Presentation creation failed: {error_msg}")
            return {
                "final_output_path": None,
                "presentation_results": {
                    "creation_successful": False,
                    "slides_created": 0,
                    "file_size": 0,
                    "template_used": bool(template_path)
                }
            }
    
    except Exception as e:
        _log_progress("📊 PRESENTATION", f"Error during presentation creation: {e}")
        return {
            "final_output_path": None,
            "presentation_results": {
                "creation_successful": False,
                "slides_created": 0,
                "file_size": 0,
                "template_used": bool(template_path)
            }
        }


def create_deck_builder_graph():
    """Create and compile the LangGraph workflow."""

    # Define workflow
    builder = StateGraph(DeckBuilderState)

    # Add nodes
    builder.add_node("research", research_node)
    builder.add_node("load_docs", document_loading_node)
    builder.add_node("create_index", indexing_node)
    builder.add_node("generate_outline", outline_generation_node)
    builder.add_node("generate_content", content_generation_node)
    builder.add_node("create_presentation", presentation_creation_node)

    # Define flow
    builder.add_edge(START, "research")
    builder.add_edge("research", "load_docs")
    builder.add_edge("load_docs", "create_index")
    builder.add_edge("create_index", "generate_outline")
    builder.add_edge("generate_outline", "generate_content")
    builder.add_edge("generate_content", "create_presentation")
    builder.add_edge("create_presentation", END)

    # Compile with memory
    memory = InMemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


# Create the global graph instance
deck_builder_graph = create_deck_builder_graph()


async def build_deck(
    user_request: str, template_path: str = None, output_path: str = None
) -> Dict[str, Any]:
    """
    Main function to build a PowerPoint deck from a user request.

    Args:
        user_request: Natural language description of the presentation needed
        template_path: Optional path to PowerPoint template
        output_path: Optional output path for the presentation

    Returns:
        Dictionary with results and status
    """

    # Initialize state
    initial_state: DeckBuilderState = {
        "messages": [
            HumanMessage(content=f"Building presentation for: {user_request}")
        ],
        "user_request": user_request,
        "search_results": [],
        "documents": [],
        "vector_index": None,
        "outline": None,
        "slide_specs": [],
        "references": [],
        "template_path": template_path,
        "output_path": output_path or "",
        "status": "Starting...",
    }

    # Run the workflow
    config = {"configurable": {"thread_id": "deck_builder_session"}}

    try:
        final_state = await deck_builder_graph.ainvoke(initial_state, config)

        return {
            "success": True,
            "output_path": final_state.get("output_path", ""),
            "slide_count": len(final_state.get("slide_specs", [])),
            "references_count": len(final_state.get("references", [])),
            "status": final_state.get("status", "Unknown"),
            "messages": [msg.content for msg in final_state.get("messages", [])],
        }

    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "status": f"Failed: {str(e)}",
        }


def build_deck_sync(
    user_request: str, template_path: str = None, output_path: str = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for build_deck function.

    Args:
        user_request: Natural language description of the presentation needed
        template_path: Optional path to PowerPoint template
        output_path: Optional output path for the presentation

    Returns:
        Dictionary with results and status
    """

    # Initialize state
    initial_state: DeckBuilderState = {
        "messages": [
            HumanMessage(content=f"Building presentation for: {user_request}")
        ],
        "user_request": user_request,
        "search_results": [],
        "documents": [],
        "vector_index": None,
        "outline": None,
        "slide_specs": [],
        "references": [],
        "template_path": template_path,
        "output_path": output_path or "",
        "status": "Starting...",
    }

    # Run the workflow synchronously
    config = {"configurable": {"thread_id": "deck_builder_session"}}

    try:
        final_state = deck_builder_graph.invoke(initial_state, config)

        output_path = final_state.get("output_path", "")
        slide_count = len(final_state.get("slide_specs", []))
        status_text = final_state.get("status", "")
        success = (not str(status_text).lower().startswith("failed")) and bool(output_path) and output_path.endswith(".pptx")

        return {
            "success": success,
            "output_path": output_path,
            "slide_count": slide_count,
            "references_count": len(final_state.get("references", [])),
            "status": final_state.get("status", "Unknown"),
            "messages": [msg.content for msg in final_state.get("messages", [])],
        }

    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "status": f"Failed: {str(e)}",
        }
