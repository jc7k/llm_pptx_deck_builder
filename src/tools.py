"""Core tools for the LLM PPTX Deck Builder with @tool decorators."""

import json
import random
import ssl
import time
from typing import List, Dict, Optional
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from pptx import Presentation
import os

from .dependencies import get_openai_llm, get_brave_search_headers
from .models import SearchResult, WebDocument
from .settings import settings
from .rate_limiter import (
    brave_limiter,
    openai_limiter,
    web_limiter,
    exponential_backoff_retry,
)

# Global storage for non-serializable objects
_vector_index_store = {}


def _create_robust_session() -> requests.Session:
    """Create a requests session with retry strategy and SSL handling."""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set reasonable timeouts and headers
    session.headers.update({
        'User-Agent': settings.user_agent or 'LLM-PPTX-Deck-Builder/1.0'
    })
    
    return session


@tool
def search_web(query: str, count: int = 10) -> List[Dict]:
    """Search web using Brave Search API for current information.

    Args:
        query: Search terms
        count: Number of results (max 20)

    Returns:
        List of search results with URLs, titles, snippets
    """
    try:
        # Rate limiting
        brave_limiter.wait_if_needed("brave_search")
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = get_brave_search_headers()

        params = {
            "q": query,
            "count": min(count, settings.max_search_results),
            "freshness": "pw",  # Past week for fresh results
            "search_lang": "en",
            "country": "US",
        }

        # Manual retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                session = _create_robust_session()
                response = session.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise e
                wait_time = (2 ** attempt) + (0.5 * random.random())
                time.sleep(wait_time)

        data = response.json()
        results = []

        for item in data.get("web", {}).get("results", []):
            result = SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("description", ""),
                published_date=item.get("age"),
            )
            results.append(result.model_dump())

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error searching web: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in web search: {e}")
        return []


@tool
def load_web_documents(urls: List[str]) -> List[Dict]:
    """Load and parse web pages using LangChain WebBaseLoader.

    Args:
        urls: List of URLs to fetch and parse

    Returns:
        List of parsed Document objects with content and metadata
    """
    try:
        documents = []

        # Limit the number of URLs to process
        urls_to_process = urls[: settings.max_documents]

        for url in urls_to_process:
            try:
                # Rate limiting for web scraping
                web_limiter.wait_if_needed("web_scraping")
                
                # Configure WebBaseLoader with SSL settings
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                loader = WebBaseLoader(
                    url,
                    requests_kwargs={
                        'verify': False,  # Disable SSL verification for problematic sites
                        'timeout': 30,
                        'headers': {
                            'User-Agent': settings.user_agent or 'LLM-PPTX-Deck-Builder/1.0'
                        }
                    }
                )
                docs = loader.load()

                for doc in docs:
                    # Skip documents with minimal content
                    if len(doc.page_content.strip()) < 100:
                        continue
                        
                    web_doc = WebDocument(
                        url=url,
                        title=doc.metadata.get("title", ""),
                        content=doc.page_content,
                        metadata=doc.metadata,
                    )
                    documents.append(web_doc.model_dump())

            except ssl.SSLError as e:
                print(f"SSL error loading document from {url}: {e}")
                continue
            except Exception as e:
                print(f"Error loading document from {url}: {e}")
                continue

        return documents

    except Exception as e:
        print(f"Error in load_web_documents: {e}")
        return []


@tool
def create_vector_index(documents: List[Dict]) -> Dict:
    """Create LlamaIndex vector store from documents.

    Args:
        documents: List of document dictionaries

    Returns:
        Index metadata for query engine creation
    """
    try:
        if not documents:
            return {"error": "No documents provided for indexing"}

        # Convert to LlamaIndex Document format
        llama_docs = []
        for doc_dict in documents:
            doc = LlamaDocument(
                text=doc_dict.get("content", ""),
                metadata={
                    "url": doc_dict.get("url", ""),
                    "title": doc_dict.get("title", ""),
                    **doc_dict.get("metadata", {}),
                },
            )
            llama_docs.append(doc)

        # Create vector index
        index = VectorStoreIndex.from_documents(llama_docs)

        metadata = {
            "document_count": len(llama_docs),
            "chunk_count": len(index.vector_store._data.embedding_dict),
            "embedding_model": settings.embedding_model,
            "created_at": datetime.now().isoformat(),
            "index_id": f"deck_builder_{int(time.time())}",
        }

        # Store index globally with unique ID
        index_id = metadata["index_id"]
        _vector_index_store[index_id] = index

        return metadata

    except Exception as e:
        print(f"Error creating vector index: {e}")
        return {"error": str(e)}


@tool
def generate_outline(topic: str, index_metadata: Dict) -> Dict:
    """Generate presentation outline using RAG.

    Args:
        topic: Presentation topic
        index_metadata: Vector index metadata

    Returns:
        JSON outline with slide titles and flow
    """
    try:
        # Retrieve index from global store
        index_id = index_metadata.get("index_id")
        if not index_id:
            return {"error": "No index ID provided"}
        
        index = _vector_index_store.get(index_id)
        if not index:
            return {"error": "Vector index not found in store"}

        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=settings.similarity_top_k)

        # Generate outline using RAG
        outline_query = f"""
        Based on the research content, create a data-driven presentation outline for: {topic}
        
        Requirements:
        1. A specific, measurable presentation objective (not generic)
        2. 8-12 slide titles that are SPECIFIC and ACTIONABLE:
           - Instead of "Introduction" → "AI Market: $500B by 2030"
           - Instead of "Challenges" → "Top 3 Adoption Barriers in 2025"
           - Instead of "Applications" → "5 Industries Transformed by AI"
           - Instead of "Future Outlook" → "2025-2030 Growth Projections"
        
        3. Each slide title should hint at the specific content it will contain
        4. Focus on insights backed by data, statistics, and real examples
        5. Target audience and their specific needs
        
        Create an outline that promises specific value, not generic topics.
        Focus on what's new, what's changing, and what actions to take.
        """

        response = query_engine.query(outline_query)

        # Use LLM to structure the response
        llm = get_openai_llm()

        structure_prompt = f"""
        Based on this research-backed content, create a JSON presentation outline:
        
        {response.response}
        
        Return ONLY valid JSON in this exact format:
        {{
            "topic": "{topic}",
            "objective": "clear presentation objective",
            "slide_titles": ["Title Slide", "Agenda", "Introduction", "..."],
            "target_audience": "target audience description",
            "duration_minutes": 15
        }}
        """

        # Rate limiting for OpenAI API
        openai_limiter.wait_if_needed("openai")
        structured_response = llm.invoke(structure_prompt)

        try:
            outline_data = json.loads(structured_response.content)
            return outline_data
        except json.JSONDecodeError:
            # Fallback outline if JSON parsing fails
            return {
                "topic": topic,
                "objective": f"Comprehensive overview of {topic}",
                "slide_titles": [
                    "Title Slide",
                    "Agenda",
                    "Introduction",
                    "Key Concepts",
                    "Current Trends",
                    "Applications",
                    "Challenges",
                    "Future Outlook",
                    "Conclusions",
                    "Next Steps",
                    "References",
                ],
                "target_audience": "General audience",
                "duration_minutes": 15,
            }

    except Exception as e:
        print(f"Error generating outline: {e}")
        return {"error": str(e)}


def validate_slide_content(slide_data: Dict) -> bool:
    """Validate slide content quality.
    
    Returns True if content meets quality standards, False otherwise.
    """
    # Check for generic filler phrases
    generic_phrases = [
        "key aspect", "important consideration", "implications and next steps",
        "key point", "main topic", "various factors", "different aspects",
        "several elements", "multiple components", "diverse range"
    ]
    
    bullets = slide_data.get("bullets", [])
    
    # Must have at least 2 bullets
    if len(bullets) < 2:
        return False
    
    # Check each bullet for quality
    for bullet in bullets:
        bullet_lower = bullet.lower()
        
        # Reject generic phrases
        if any(phrase in bullet_lower for phrase in generic_phrases):
            return False
        
        # Should contain at least one number, percentage, or specific term
        has_specifics = (
            any(char.isdigit() for char in bullet) or
            "%" in bullet or
            "$" in bullet or
            len(bullet.split()) > 5  # Substantial content
        )
        
        if not has_specifics and len(bullet) < 30:
            return False
    
    return True


@tool
def generate_slide_content(outline: Dict, index_metadata: Dict) -> List[Dict]:
    """Generate detailed slide content with citations.

    Args:
        outline: Presentation outline
        index_metadata: Vector index for RAG queries

    Returns:
        List of slide specifications with bullets, notes, citations
    """
    try:
        # Retrieve index from global store
        index_id = index_metadata.get("index_id")
        if not index_id:
            return [{"error": "No index ID provided"}]
        
        index = _vector_index_store.get(index_id)
        if not index:
            return [{"error": "Vector index not found in store"}]

        query_engine = index.as_query_engine(similarity_top_k=settings.similarity_top_k)

        slides = []
        slide_titles = outline.get("slide_titles", [])

        for i, title in enumerate(slide_titles):
            # Skip title and agenda slides - handle them separately
            if i == 0 or title.lower() in ["agenda", "references"]:
                continue

            try:
                # Generate content for this slide using RAG
                # Create specialized queries based on slide type
                title_lower = title.lower()
                
                if "introduction" in title_lower or "overview" in title_lower:
                    content_query = f"""
                    For the introduction slide about {outline.get("topic", "")}, provide:
                    - Current state statistics and market size
                    - Key problem statement or opportunity with data
                    - Most significant recent development or trend
                    - Primary stakeholders affected and impact metrics
                    Include specific numbers, dates, and quantifiable impacts.
                    """
                elif "trend" in title_lower or "current" in title_lower:
                    content_query = f"""
                    For current trends in {outline.get("topic", "")}, provide:
                    - Top 3-4 emerging trends with growth percentages
                    - Timeline of recent developments (last 2 years)
                    - Market adoption rates or user statistics
                    - Leading companies or initiatives with specific examples
                    Focus on data from 2024-2025 with concrete metrics.
                    """
                elif "application" in title_lower or "use case" in title_lower:
                    content_query = f"""
                    For real-world applications of {outline.get("topic", "")}, provide:
                    - Specific industry implementations with company names
                    - Measurable outcomes or ROI from deployments
                    - Technical specifications or performance metrics
                    - Success stories with quantified results
                    Include concrete examples and case studies.
                    """
                elif "challenge" in title_lower or "problem" in title_lower or "risk" in title_lower:
                    content_query = f"""
                    For challenges related to {outline.get("topic", "")}, provide:
                    - Top obstacles with impact percentages or costs
                    - Technical limitations with specific metrics
                    - Regulatory or compliance issues with deadlines
                    - Mitigation strategies with success rates
                    Include quantified impacts and specific examples.
                    """
                elif "future" in title_lower or "outlook" in title_lower or "prediction" in title_lower:
                    content_query = f"""
                    For future outlook of {outline.get("topic", "")}, provide:
                    - Market projections with specific years and values
                    - Upcoming technologies or innovations with timelines
                    - Expected adoption rates or growth percentages
                    - Key milestones or developments to watch
                    Include forecasts from authoritative sources with dates.
                    """
                elif "conclusion" in title_lower or "summary" in title_lower:
                    content_query = f"""
                    For conclusions about {outline.get("topic", "")}, provide:
                    - Key takeaways with supporting statistics
                    - Most important finding or insight with evidence
                    - Strategic implications with timeframes
                    - Critical success factors with metrics
                    Synthesize the most impactful points with data.
                    """
                elif "next step" in title_lower or "action" in title_lower or "recommendation" in title_lower:
                    content_query = f"""
                    For actionable next steps regarding {outline.get("topic", "")}, provide:
                    - Immediate actions with timelines (30/60/90 days)
                    - Required resources or investments with estimates
                    - Success metrics or KPIs to track
                    - Risk mitigation steps with priorities
                    Include specific, measurable action items.
                    """
                else:
                    content_query = f"""
                    For a slide titled "{title}" about {outline.get("topic", "")}, provide:
                    - Specific statistics, percentages, or metrics
                    - Concrete examples with names and dates
                    - Quantifiable impacts or outcomes
                    - Evidence-based insights from authoritative sources
                    NO generic statements - only specific, factual content.
                    """

                response = query_engine.query(content_query)

                # Structure the content using LLM
                llm = get_openai_llm()

                structure_prompt = f"""
                Based on this research content, create structured slide content:
                
                {response.response}
                
                You MUST create substantive, fact-based content for the slide titled "{title}".
                
                REQUIREMENTS:
                - Each bullet point MUST contain a specific fact, statistic, or actionable insight
                - NO generic statements like "Key aspects" or "Important considerations"
                - Include specific numbers, percentages, dates, or concrete examples
                - Speaker notes must explain the significance and provide context
                
                Return ONLY valid JSON in this exact format:
                {{
                    "title": "{title}",
                    "bullets": [
                        "Specific fact with data point or concrete example",
                        "Another specific insight or statistic",
                        "Actionable recommendation or key finding"
                    ],
                    "speaker_notes": "Detailed explanation with context and citations [1], [2]",
                    "slide_type": "content"
                }}
                
                Generate 3-5 specific, substantive bullet points. No filler content.
                """

                # Rate limiting for OpenAI API
                openai_limiter.wait_if_needed("openai")
                structured_response = llm.invoke(structure_prompt)

                try:
                    slide_data = json.loads(structured_response.content)

                    # Extract references from the response
                    references = []
                    for source_node in response.source_nodes:
                        if hasattr(source_node, "node") and hasattr(
                            source_node.node, "metadata"
                        ):
                            url = source_node.node.metadata.get("url")
                            if url and url not in references:
                                references.append(url)

                    slide_data["references"] = references
                    
                    # Validate content quality
                    if validate_slide_content(slide_data):
                        slides.append(slide_data)
                    else:
                        print(f"Content quality check failed for {title}, regenerating...")
                        raise json.JSONDecodeError("Quality check failed", "", 0)

                except json.JSONDecodeError:
                    # Retry with simpler prompt
                    print(f"JSON parsing failed for {title}, retrying with simpler prompt...")
                    
                    retry_prompt = f"""
                    Create content for a slide about "{title}".
                    Based on: {response.response[:1000]}
                    
                    Provide exactly 3 bullet points with specific facts.
                    Format as JSON:
                    {{"title": "{title}", "bullets": ["fact1", "fact2", "fact3"], "speaker_notes": "explanation"}}
                    """
                    
                    openai_limiter.wait_if_needed("openai")
                    retry_response = llm.invoke(retry_prompt)
                    
                    try:
                        slide_data = json.loads(retry_response.content)
                        slide_data["slide_type"] = "content"
                        slide_data["references"] = references
                        slides.append(slide_data)
                    except:
                        # Skip this slide rather than add filler
                        print(f"Could not generate quality content for {title}, skipping...")
                        continue

            except Exception as e:
                print(f"Error generating content for slide '{title}': {e}")
                continue

        return slides

    except Exception as e:
        print(f"Error in generate_slide_content: {e}")
        return []


@tool
def create_presentation(
    slide_specs: List[Dict], template_path: Optional[str] = None
) -> str:
    """Generate PowerPoint file using python-pptx.

    Args:
        slide_specs: List of slide content dictionaries
        template_path: Optional PPTX template file path

    Returns:
        Path to generated PPTX file
    """
    try:
        # Create presentation from template or blank
        if template_path and os.path.exists(template_path):
            prs = Presentation(template_path)
        else:
            prs = Presentation()

        # Get the first slide spec to determine topic
        topic = "Presentation"
        if slide_specs:
            topic = slide_specs[0].get("title", "Presentation")

        # Create title slide
        title_slide = prs.slides.add_slide(prs.slide_layouts[0])
        title_slide.shapes.title.text = topic
        subtitle = title_slide.placeholders[1]
        subtitle.text = f"Generated on {datetime.now().strftime('%B %d, %Y')}"

        # Create agenda slide
        agenda_slide = prs.slides.add_slide(prs.slide_layouts[1])
        agenda_slide.shapes.title.text = "Agenda"

        if len(slide_specs) > 0:
            agenda_content = agenda_slide.placeholders[1]
            agenda_items = [
                spec.get("title", "") for spec in slide_specs[:6]
            ]  # First 6 items
            agenda_content.text = "\n".join(
                [item for item in agenda_items if item]
            )

        # Create content slides
        for spec in slide_specs:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = spec.get("title", "")

            # Add bullet points
            bullets = spec.get("bullets", [])
            if bullets:
                content = slide.placeholders[1]
                content.text = "\n".join([bullet for bullet in bullets])

            # Add speaker notes
            notes = spec.get("speaker_notes", "")
            if notes:
                notes_slide = slide.notes_slide
                notes_slide.notes_text_frame.text = notes

        # Create references slide
        all_references = []
        for spec in slide_specs:
            refs = spec.get("references", [])
            all_references.extend(refs)

        # Remove duplicates while preserving order
        unique_refs = []
        for ref in all_references:
            if ref and ref not in unique_refs:
                unique_refs.append(ref)

        if unique_refs:
            ref_slide = prs.slides.add_slide(prs.slide_layouts[1])
            ref_slide.shapes.title.text = "References"

            ref_content = ref_slide.placeholders[1]
            ref_text = "\n".join(
                [f"[{i+1}] {ref}" for i, ref in enumerate(unique_refs)]
            )
            ref_content.text = ref_text

        # Save presentation
        output_dir = settings.default_output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(
            c for c in topic if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        filename = f"{safe_topic}_{timestamp}.pptx"
        output_path = os.path.join(output_dir, filename)

        prs.save(output_path)

        return output_path

    except Exception as e:
        print(f"Error creating presentation: {e}")
        return f"Error: {str(e)}"


# Additional utility tools


@tool
def extract_urls_from_search_results(search_results: List[Dict]) -> List[str]:
    """Extract URLs from search results for document loading.

    Args:
        search_results: List of search result dictionaries

    Returns:
        List of URLs to load
    """
    urls = []
    for result in search_results:
        url = result.get("url", "")
        if url and url.startswith(("http://", "https://")):
            urls.append(url)
    return urls


@tool
def deduplicate_references(references: List[str]) -> List[str]:
    """Remove duplicate references while preserving order.

    Args:
        references: List of reference URLs

    Returns:
        Deduplicated list of references
    """
    seen = set()
    deduped = []
    for ref in references:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped
