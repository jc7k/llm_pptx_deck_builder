"""Main LangGraph workflow for the LLM PPTX Deck Builder."""

import time
from typing import Dict, Any, Optional, Callable
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .models import DeckBuilderState
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

        if search_results:
            urls = [result.get("url", "") for result in search_results[:5]]
            _log_progress("🔍 RESEARCH", f"Top sources: {', '.join(urls)}")

        # Add a human message about the research
        research_message = HumanMessage(
            content=f"Found {len(search_results)} search results for: {user_request}"
        )

        return {
            "search_results": search_results,
            "messages": state["messages"] + [research_message],
            "status": "Research completed",
        }

    except Exception as e:
        _log_progress("🔍 RESEARCH", f"❌ Research failed: {str(e)}")
        error_message = HumanMessage(content=f"Research failed: {str(e)}")
        return {
            "search_results": [],
            "messages": state["messages"] + [error_message],
            "status": f"Research failed: {str(e)}",
        }


def document_loading_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Document loading node: Load web pages using LangChain WebBaseLoader."""
    try:
        search_results = state["search_results"]

        _log_progress(
            "📄 DOCUMENTS", f"Processing {len(search_results)} search results..."
        )

        # Update status
        state["status"] = "Loading documents..."

        # Extract URLs from search results
        _log_progress("📄 DOCUMENTS", "Extracting URLs from search results...")
        urls = extract_urls_from_search_results.invoke(
            {"search_results": search_results}
        )

        _log_progress("📄 DOCUMENTS", f"Found {len(urls)} URLs to process")
        _log_progress("📄 DOCUMENTS", "Starting document loading with rate limiting...")

        # Load documents
        documents = load_web_documents.invoke({"urls": urls})

        _log_progress(
            "📄 DOCUMENTS", f"✅ Successfully loaded {len(documents)} documents"
        )

        # Log document details
        if documents:
            total_chars = sum(len(doc.get("content", "")) for doc in documents)
            _log_progress("📄 DOCUMENTS", f"Total content: {total_chars:,} characters")

        loading_message = HumanMessage(
            content=f"Loaded {len(documents)} documents from {len(urls)} URLs"
        )

        return {
            "documents": documents,
            "messages": state["messages"] + [loading_message],
            "status": "Document loading completed",
        }

    except Exception as e:
        _log_progress("📄 DOCUMENTS", f"❌ Document loading failed: {str(e)}")
        error_message = HumanMessage(content=f"Document loading failed: {str(e)}")
        return {
            "documents": [],
            "messages": state["messages"] + [error_message],
            "status": f"Document loading failed: {str(e)}",
        }


def indexing_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Indexing node: Create LlamaIndex vector store."""
    try:
        documents = state["documents"]

        _log_progress(
            "🧠 INDEXING", f"Creating vector index from {len(documents)} documents..."
        )
        _log_progress("🧠 INDEXING", "Converting documents to LlamaIndex format...")

        # Update status
        state["status"] = "Creating vector index..."

        # Create vector index
        _log_progress("🧠 INDEXING", "Building vector embeddings with OpenAI...")
        vector_index = create_vector_index.invoke({"documents": documents})

        if "error" in vector_index:
            raise Exception(vector_index["error"])

        doc_count = vector_index.get("document_count", 0)
        chunk_count = vector_index.get("chunk_count", 0)
        _log_progress(
            "🧠 INDEXING",
            f"✅ Vector index created: {doc_count} docs, {chunk_count} chunks",
        )
        _log_progress(
            "🧠 INDEXING", f"Index ID: {vector_index.get('index_id', 'unknown')}"
        )

        indexing_message = HumanMessage(
            content=f"Created vector index with {doc_count} documents"
        )

        return {
            "vector_index": vector_index,
            "messages": state["messages"] + [indexing_message],
            "status": "Vector indexing completed",
        }

    except Exception as e:
        _log_progress("🧠 INDEXING", f"❌ Indexing failed: {str(e)}")
        error_message = HumanMessage(content=f"Indexing failed: {str(e)}")
        return {
            "vector_index": None,
            "messages": state["messages"] + [error_message],
            "status": f"Indexing failed: {str(e)}",
        }


def outline_generation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Outline generation node: Create presentation outline using RAG."""
    try:
        user_request = state["user_request"]
        vector_index = state["vector_index"]

        _log_progress(
            "📋 OUTLINE", f"Generating presentation outline for: {user_request}"
        )
        _log_progress("📋 OUTLINE", "Querying vector index for relevant content...")

        # Update status
        state["status"] = "Generating presentation outline..."

        # Generate outline
        _log_progress("📋 OUTLINE", "Applying rate limiting for OpenAI API...")
        outline = generate_outline.invoke(
            {"topic": user_request, "index_metadata": vector_index}
        )

        if "error" in outline:
            raise Exception(outline["error"])

        slide_count = len(outline.get("slide_titles", []))
        _log_progress("📋 OUTLINE", f"✅ Generated outline with {slide_count} slides")

        # Log slide titles
        if outline.get("slide_titles"):
            titles = outline["slide_titles"][:5]  # Show first 5
            _log_progress(
                "📋 OUTLINE",
                f"Slides: {', '.join(titles)}{'...' if len(outline['slide_titles']) > 5 else ''}",
            )

        duration = outline.get("duration_minutes", "Unknown")
        _log_progress("📋 OUTLINE", f"Estimated duration: {duration} minutes")

        outline_message = HumanMessage(
            content=f"Generated outline with {slide_count} slides for: {outline.get('topic', user_request)}"
        )

        return {
            "outline": outline,
            "messages": state["messages"] + [outline_message],
            "status": "Outline generation completed",
        }

    except Exception as e:
        error_message = HumanMessage(content=f"Outline generation failed: {str(e)}")
        return {
            "outline": None,
            "messages": state["messages"] + [error_message],
            "status": f"Outline generation failed: {str(e)}",
        }


def content_generation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Content generation node: Generate detailed slide content with citations."""
    try:
        outline = state["outline"]
        vector_index = state["vector_index"]

        slide_count = len(outline.get("slide_titles", []))
        _log_progress(
            "📝 CONTENT", f"Generating detailed content for {slide_count} slides..."
        )
        _log_progress(
            "📝 CONTENT", "Querying vector index for slide-specific content..."
        )

        # Update status
        state["status"] = "Generating slide content..."

        # Generate slide content
        _log_progress("📝 CONTENT", "Applying rate limiting for OpenAI API...")
        _log_progress(
            "📝 CONTENT", "Processing slides with RAG-based content generation..."
        )

        slide_specs = generate_slide_content.invoke(
            {"outline": outline, "index_metadata": vector_index}
        )

        if not slide_specs or any("error" in spec for spec in slide_specs):
            raise Exception("Failed to generate slide content")

        _log_progress(
            "📝 CONTENT", f"✅ Generated content for {len(slide_specs)} slides"
        )

        # Collect all references
        _log_progress("📝 CONTENT", "Collecting and processing citations...")
        all_references = []
        for i, spec in enumerate(slide_specs, 1):
            refs = spec.get("references", [])
            all_references.extend(refs)
            if refs:
                _log_progress("📝 CONTENT", f"Slide {i}: {len(refs)} citations")

        _log_progress(
            "📝 CONTENT", f"Total citations before deduplication: {len(all_references)}"
        )

        # Deduplicate references
        _log_progress("📝 CONTENT", "Deduplicating citations...")
        unique_references = deduplicate_references.invoke(
            {"references": all_references}
        )

        _log_progress(
            "📝 CONTENT", f"✅ Final unique citations: {len(unique_references)}"
        )

        # Log content summary
        total_bullets = sum(len(spec.get("bullets", [])) for spec in slide_specs)
        _log_progress(
            "📝 CONTENT",
            f"Generated {total_bullets} total bullet points across all slides",
        )

        content_message = HumanMessage(
            content=f"Generated content for {len(slide_specs)} slides with {len(unique_references)} unique references"
        )

        return {
            "slide_specs": slide_specs,
            "references": unique_references,
            "messages": state["messages"] + [content_message],
            "status": "Content generation completed",
        }

    except Exception as e:
        _log_progress("📝 CONTENT", f"❌ Content generation failed: {str(e)}")
        error_message = HumanMessage(content=f"Content generation failed: {str(e)}")
        return {
            "slide_specs": [],
            "references": [],
            "messages": state["messages"] + [error_message],
            "status": f"Content generation failed: {str(e)}",
        }


def presentation_creation_node(state: DeckBuilderState) -> Dict[str, Any]:
    """Presentation creation node: Generate PowerPoint file using python-pptx."""
    try:
        slide_specs = state["slide_specs"]
        template_path = state.get("template_path")
        references = state.get("references", [])

        slide_count = len(slide_specs)
        _log_progress(
            "🎨 PRESENTATION", f"Creating PowerPoint file with {slide_count} slides..."
        )

        if slide_count == 0:
            error_message = HumanMessage(content="No slides to render")
            return {
                "output_path": "",
                "messages": state["messages"] + [error_message],
                "status": "Presentation creation failed: No slides to render",
            }

        if template_path:
            _log_progress("🎨 PRESENTATION", f"Using template: {template_path}")
        else:
            _log_progress("🎨 PRESENTATION", "Using default PowerPoint template")

        _log_progress("🎨 PRESENTATION", f"Including {len(references)} references")

        # Update status
        state["status"] = "Creating PowerPoint presentation..."

        # Create presentation
        _log_progress("🎨 PRESENTATION", "Initializing python-pptx presentation...")
        _log_progress("🎨 PRESENTATION", "Rendering title slide...")
        _log_progress("🎨 PRESENTATION", "Processing content slides...")

        output_path = create_presentation.invoke(
            {"slide_specs": slide_specs, "template_path": template_path}
        )

        if output_path.startswith("Error:"):
            raise Exception(output_path)

        _log_progress("🎨 PRESENTATION", f"✅ Presentation saved to: {output_path}")

        # Calculate file size if possible
        try:
            import os

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                file_size_mb = file_size / (1024 * 1024)
                _log_progress("🎨 PRESENTATION", f"File size: {file_size_mb:.1f} MB")
        except Exception:
            pass  # File size calculation is optional

        _log_progress("🎨 PRESENTATION", "🎉 Presentation generation complete!")

        presentation_message = HumanMessage(
            content=f"Successfully created presentation: {output_path}"
        )

        return {
            "output_path": output_path,
            "messages": state["messages"] + [presentation_message],
            "status": "Presentation creation completed",
        }

    except Exception as e:
        _log_progress("🎨 PRESENTATION", f"❌ Presentation creation failed: {str(e)}")
        error_message = HumanMessage(content=f"Presentation creation failed: {str(e)}")
        return {
            "output_path": "",
            "messages": state["messages"] + [error_message],
            "status": f"Presentation creation failed: {str(e)}",
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
