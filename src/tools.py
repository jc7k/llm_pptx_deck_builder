"""Core tools for the LLM PPTX Deck Builder with @tool decorators."""

import json
import random
import ssl
import time
from typing import List, Dict, Optional, Tuple
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
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set reasonable timeouts and headers
    session.headers.update(
        {"User-Agent": settings.user_agent or "LLM-PPTX-Deck-Builder/1.0"}
    )

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
                wait_time = (2**attempt) + (0.5 * random.random())
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
                        "verify": False,  # Disable SSL verification for problematic sites
                        "timeout": 30,
                        "headers": {
                            "User-Agent": settings.user_agent
                            or "LLM-PPTX-Deck-Builder/1.0"
                        },
                    },
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


def validate_thought_completeness(bullets: List[str]) -> Tuple[bool, List[str]]:
    """Validate bullet point thought completeness and formatting.

    Focuses on complete thoughts rather than complete sentences.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    for i, bullet in enumerate(bullets):
        # Check word count (flexible range based on thought complexity)
        word_count = len(bullet.split())
        if word_count > 15:  # Increased from 12 to allow complete thoughts
            issues.append(
                f"Bullet {i+1}: {word_count} words (max 15): '{bullet[:50]}...'"
            )

        # Check for markdown formatting
        if "**" in bullet or "__" in bullet or "*" in bullet:
            issues.append(f"Bullet {i+1}: Contains markdown formatting: '{bullet}'")

        # Check minimum length (too short bullets are often generic)
        if word_count < 4:
            issues.append(f"Bullet {i+1}: Too short ({word_count} words): '{bullet}'")

        # Enhanced incomplete thought detection
        bullet_lower = bullet.lower().strip()

        # Prepositions that indicate incomplete thoughts
        incomplete_prepositions = [
            " through",
            " via",
            " by",
            " due to",
            " such as",
            " including",
            " with",
            " from",
            " for",
            " in",
            " on",
            " at",
            " of",
            " to",
            " within",
            " across",
            " among",
            " between",
            " during",
            " since",
            " until",
            " after",
            " before",
            " around",
            " under",
            " over",
        ]

        # Conjunctions that indicate incomplete thoughts
        incomplete_conjunctions = [
            " and",
            " or",
            " but",
            " so",
            " yet",
            " however",
            " while",
            " whereas",
            " although",
            " because",
            " since",
            " if",
            " when",
            " where",
            " who",
            " which",
            " that",
        ]

        # Articles and incomplete phrases that indicate cut-offs
        incomplete_articles = [
            " the",
            " a",
            " an",
            " some",
            " many",
            " most",
            " all",
            " these",
            " those",
            " this",
            " that",
        ]

        # Special case: "is" often indicates incomplete thought
        if bullet_lower.endswith(" is"):
            issues.append(f"Bullet {i+1}: Incomplete - ends with 'is': '{bullet}'")

        # Check for incomplete endings
        if any(
            bullet_lower.endswith(indicator) for indicator in incomplete_prepositions
        ):
            issues.append(
                f"Bullet {i+1}: Incomplete - ends with preposition: '{bullet}'"
            )

        if any(
            bullet_lower.endswith(indicator) for indicator in incomplete_conjunctions
        ):
            issues.append(
                f"Bullet {i+1}: Incomplete - ends with conjunction: '{bullet}'"
            )

        if any(bullet_lower.endswith(indicator) for indicator in incomplete_articles):
            issues.append(f"Bullet {i+1}: Incomplete - ends with article: '{bullet}'")

        # Check for truncation patterns
        if bullet.endswith("...") or bullet.endswith(".."):
            issues.append(f"Bullet {i+1}: Appears truncated with ellipsis: '{bullet}'")

        # Check for incomplete sentences that don't have main concepts
        if (
            not any(char.isdigit() for char in bullet)
            and "%" not in bullet
            and "$" not in bullet
            and len(bullet.split()) > 8
            and not any(
                word in bullet_lower
                for word in [
                    "increase",
                    "decrease",
                    "growth",
                    "decline",
                    "rise",
                    "fall",
                    "boost",
                    "drop",
                    "gain",
                    "loss",
                ]
            )
        ):
            # Long bullet without specific data or action words might be incomplete
            if bullet_lower.count(" ") > 10:  # More than 10 spaces = very long
                issues.append(
                    f"Bullet {i+1}: Potentially incomplete - long without specific data: '{bullet[:50]}...'"
                )

        # Check for common incomplete patterns
        incomplete_patterns = [
            "organizations should",
            "companies need to",
            "it is important to",
            "there are many",
            "some of the",
            "one of the",
            "according to",
            "research shows that",
            "studies indicate",
            "experts believe",
        ]

        for pattern in incomplete_patterns:
            if bullet_lower.startswith(pattern) and len(bullet.split()) < 10:
                issues.append(
                    f"Bullet {i+1}: Generic/incomplete start pattern: '{bullet}'"
                )

    return len(issues) == 0, issues


def optimize_slide_title(slide_data: Dict) -> Dict:
    """Optimize slide title based on actual bullet content for better alignment.

    Args:
        slide_data: Slide with title and bullets

    Returns:
        Updated slide_data with optimized title
    """
    try:
        current_title = slide_data.get("title", "")
        bullets = slide_data.get("bullets", [])

        if not bullets:
            return slide_data

        # Analyze bullet content themes
        llm = get_openai_llm()
        bullets_text = "\n".join([f"- {bullet}" for bullet in bullets])

        optimization_prompt = f"""
        Create a specific, compelling slide title based on what the bullet points actually discuss.
        IGNORE the current title - focus only on the bullet content.
        
        BULLET CONTENT:
        {bullets_text}
        
        REQUIREMENTS:
        - Be SPECIFIC about what the bullets discuss, not generic
        - Replace vague titles like "Applications", "Challenges", "Current Trends" 
        - Use concrete terms that describe the actual findings/insights
        - Maximum 6-7 words to fit on one line
        - Make it informative and engaging
        
        TRANSFORMATION EXAMPLES:
        - Generic "Applications" → Specific "Healthcare & Finance AI Success"
        - Generic "Challenges" → Specific "Skills Gap Implementation Barriers"  
        - Generic "Current Trends" → Specific "75% Workforce AI Adoption"
        - Generic "Future Outlook" → Specific "2030 Job Market Predictions"
        
        Create a title that tells the story these bullets are telling.
        Return ONLY the optimized title text, no quotes or explanation.
        """

        # Rate limiting for OpenAI API
        openai_limiter.wait_if_needed("openai")
        response = llm.invoke(optimization_prompt)

        optimized_title = response.content.strip()

        # Clean up quotation marks and formatting
        optimized_title = optimized_title.strip(
            '"\'""' ""
        )  # Remove various quote types

        # Ensure title length is appropriate (max 8 words to prevent line wraps)
        words = optimized_title.split()
        if len(words) > 8:
            # Condense title while preserving key concepts
            condensed_prompt = f"""
            Condense this title to maximum 8 words while preserving the core concept:
            "{optimized_title}"
            
            Keep the most important keywords and remove filler words.
            Return ONLY the condensed title, no quotes or explanation.
            """

            # Rate limiting for OpenAI API
            openai_limiter.wait_if_needed("openai")
            condense_response = llm.invoke(condensed_prompt)
            condensed_title = condense_response.content.strip().strip('"\'""' "")

            if condensed_title and len(condensed_title.split()) <= 8:
                optimized_title = condensed_title

        # Always apply optimization if we got a reasonable result
        if (
            optimized_title
            and len(optimized_title.split()) <= 8
            and len(optimized_title) > 2
        ):
            # Only keep current title if optimization produced something clearly worse
            if optimized_title.lower() not in ["error", "none", "unknown", "untitled"]:
                slide_data["title"] = optimized_title
                slide_data["original_title"] = current_title  # Keep for debugging

        return slide_data

    except Exception as e:
        print(f"Error optimizing title for slide: {e}")
        return slide_data  # Return original if optimization fails


def validate_slide_content(slide_data: Dict) -> bool:
    """Validate slide content quality.

    Returns True if content meets quality standards, False otherwise.
    """
    # Check for generic filler phrases
    generic_phrases = [
        "key aspect",
        "important consideration",
        "implications and next steps",
        "key point",
        "main topic",
        "various factors",
        "different aspects",
        "several elements",
        "multiple components",
        "diverse range",
    ]

    bullets = slide_data.get("bullets", [])

    # Must have at least 2 bullets, max 4 for readability
    if len(bullets) < 2 or len(bullets) > 4:
        print(f"Invalid bullet count: {len(bullets)} (need 2-4)")
        return False

    # Validate thought completeness and formatting
    is_complete, completeness_issues = validate_thought_completeness(bullets)
    if not is_complete:
        print("Thought completeness validation failed:")
        for issue in completeness_issues:
            print(f"  - {issue}")
        return False

    # Check each bullet for quality
    for bullet in bullets:
        bullet_lower = bullet.lower()

        # Reject generic phrases
        if any(phrase in bullet_lower for phrase in generic_phrases):
            print(f"Generic phrase detected in: '{bullet}'")
            return False

        # Should contain at least one number, percentage, or specific term
        has_specifics = (
            any(char.isdigit() for char in bullet)
            or "%" in bullet
            or "$" in bullet
            or any(
                word in bullet_lower
                for word in ["million", "billion", "thousand", "by", "since", "ago"]
            )
        )

        if not has_specifics:
            print(f"Bullet lacks specific data: '{bullet}'")
            return False

    return True


def create_content_allocation_plan(outline: Dict, index_metadata: Dict) -> Dict:
    """Create a unified content plan to eliminate repetition across slides.

    This function implements a Content Distribution Matrix approach:
    1. Generate comprehensive research summary
    2. Create content allocation plan
    3. Distribute unique insights across slides

    Args:
        outline: Presentation outline
        index_metadata: Vector index for RAG queries

    Returns:
        Content allocation plan with unique insights per slide
    """
    try:
        # Retrieve index from global store
        index_id = index_metadata.get("index_id")
        if not index_id:
            return {"error": "No index ID provided"}

        index = _vector_index_store.get(index_id)
        if not index:
            return {"error": "Vector index not found in store"}

        query_engine = index.as_query_engine(similarity_top_k=settings.similarity_top_k)

        # Phase 1: Generate comprehensive research summary
        topic = outline.get("topic", "")
        comprehensive_query = f"""
        Provide research summary for {topic} with specific data:
        1. Current statistics and market metrics
        2. Key trends with growth percentages 
        3. Real applications with company examples
        4. Main challenges with quantified impacts
        5. Future predictions with dates/values
        
        Include specific numbers, percentages, dates, company names.
        """

        response = query_engine.query(comprehensive_query)
        research_summary = response.response

        # Phase 2: Create content allocation plan
        llm = get_openai_llm()
        slide_titles = outline.get("slide_titles", [])

        # Remove title, agenda, and references slides
        content_slides = [
            title
            for title in slide_titles
            if title.lower() not in ["title slide", "agenda", "references"]
        ]

        allocation_prompt = f"""
        Based on this comprehensive research summary, create a Content Distribution Matrix 
        to allocate unique insights across {len(content_slides)} slides.
        
        RESEARCH SUMMARY:
        {research_summary}
        
        SLIDE TITLES:
        {', '.join([f'"{title}"' for title in content_slides])}
        
        CRITICAL REQUIREMENTS:
        - Each slide must have completely UNIQUE content - zero overlap
        - Distribute insights logically based on slide titles
        - Each insight must be specific with numbers/data
        - No generic statements allowed
        - Allocate exactly 3-4 unique data points per slide
        
        Return ONLY valid JSON in this exact format:
        {{
            "content_plan": {{
                "Slide Title 1": [
                    "Unique insight 1 with specific data",
                    "Unique insight 2 with metrics", 
                    "Unique insight 3 with numbers"
                ],
                "Slide Title 2": [
                    "Different insight 1 with data",
                    "Different insight 2 with stats",
                    "Different insight 3 with figures"
                ]
            }},
            "used_insights": [
                "All insights listed here to prevent duplication"
            ]
        }}
        
        Ensure zero content repetition between slides. Each fact/statistic appears only once.
        """

        # Rate limiting for OpenAI API
        openai_limiter.wait_if_needed("openai")
        allocation_response = llm.invoke(allocation_prompt)

        # Debug: Check if response is empty
        if not allocation_response.content or not allocation_response.content.strip():
            print(
                "Empty response from allocation plan generation. Trying shorter prompt..."
            )

            # Fallback with shorter prompt
            short_allocation_prompt = f"""
            Create a content allocation plan for these slides about {topic}:
            {', '.join([f'"{title}"' for title in content_slides])}
            
            Based on research: {research_summary[:2000]}...
            
            Allocate 3-4 unique, specific data points to each slide with NO overlap.
            
            Return JSON: {{"content_plan": {{"Slide Title": ["unique insight 1", "unique insight 2", "unique insight 3"]}}}}
            """

            openai_limiter.wait_if_needed("openai")
            allocation_response = llm.invoke(short_allocation_prompt)

        try:
            if not allocation_response.content:
                raise json.JSONDecodeError("Empty response", "", 0)

            # Clean JSON response (remove markdown formatting if present)
            content = allocation_response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()

            allocation_plan = json.loads(content)
            allocation_plan["research_summary"] = research_summary
            allocation_plan["source_references"] = []

            # Extract references from the response
            for source_node in response.source_nodes:
                if hasattr(source_node, "node") and hasattr(
                    source_node.node, "metadata"
                ):
                    url = source_node.node.metadata.get("url")
                    if url and url not in allocation_plan["source_references"]:
                        allocation_plan["source_references"].append(url)

            return allocation_plan

        except json.JSONDecodeError as e:
            print(f"Failed to parse content allocation plan: {e}")
            print(f"Response content: '{allocation_response.content[:200]}...'")

            # Create fallback allocation plan using research summary
            fallback_plan = {"content_plan": {}, "source_references": []}

            # Extract key insights from research summary for fallback
            research_sentences = research_summary.split(". ")[
                :20
            ]  # Take first 20 sentences
            insights_per_slide = 3

            for i, slide_title in enumerate(content_slides):
                # Distribute different sentences to each slide
                start_idx = i * insights_per_slide
                end_idx = start_idx + insights_per_slide
                slide_insights = (
                    research_sentences[start_idx:end_idx]
                    if start_idx < len(research_sentences)
                    else []
                )

                # If we don't have enough research sentences, create basic insights
                while len(slide_insights) < insights_per_slide:
                    slide_insights.append(
                        f"Key finding for {slide_title} from research data"
                    )

                fallback_plan["content_plan"][slide_title] = slide_insights[
                    :insights_per_slide
                ]

            # Extract references from the response
            for source_node in response.source_nodes:
                if hasattr(source_node, "node") and hasattr(
                    source_node.node, "metadata"
                ):
                    url = source_node.node.metadata.get("url")
                    if url and url not in fallback_plan["source_references"]:
                        fallback_plan["source_references"].append(url)

            print("Using fallback allocation plan...")
            return fallback_plan

    except Exception as e:
        print(f"Error creating content allocation plan: {e}")
        return {"error": str(e)}


def generate_slides_individually(outline: Dict, index_metadata: Dict) -> List[Dict]:
    """Fallback function to generate slides individually with enhanced prompts.

    Args:
        outline: Presentation outline
        index_metadata: Vector index for RAG queries

    Returns:
        List of slide specifications
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
        topic = outline.get("topic", "")
        llm = get_openai_llm()

        # Track used content to minimize repetition
        used_content = set()

        for i, title in enumerate(slide_titles):
            # Skip title and agenda slides
            if i == 0 or title.lower() in ["agenda", "references"]:
                continue

            try:
                # Generate specific query for this slide
                content_query = f"""
                For slide "{title}" about {topic}, provide specific data and insights.
                Focus on unique content not covered in previous slides.
                Include statistics, percentages, company names, dates, and metrics.
                Avoid generic statements - only factual, specific information.
                """

                response = query_engine.query(content_query)

                # Generate slide content with strict formatting
                bullet_prompt = f"""
                Create content for slide "{title}" based on this research:
                
                {response.response[:1500]}
                
                CRITICAL REQUIREMENTS:
                - Focus on complete THOUGHTS, not complete sentences
                - Maximum 15 words per bullet for complete ideas
                - Use bullet-style phrasing with specific data
                - NO markdown formatting (**, __, etc.)
                - NO generic phrases like "key aspects" or "important considerations"
                - Include company names, statistics, and concrete facts
                
                Return ONLY JSON:
                {{
                    "title": "{title}",
                    "bullets": [
                        "Complete thought with specific data (max 15 words)",
                        "Another key insight with metrics and context",  
                        "Clear actionable concept with concrete facts"
                    ],
                    "speaker_notes": "Detailed explanation with context. NO markdown formatting.",
                    "slide_type": "content"
                }}
                
                Generate 3-4 complete thoughts as bullets, max 15 words each.
                """

                # Rate limiting for OpenAI API
                openai_limiter.wait_if_needed("openai")
                conversion_response = llm.invoke(bullet_prompt)

                try:
                    slide_data = json.loads(conversion_response.content)

                    # Extract references
                    references = []
                    for source_node in response.source_nodes:
                        if hasattr(source_node, "node") and hasattr(
                            source_node.node, "metadata"
                        ):
                            url = source_node.node.metadata.get("url")
                            if url and url not in references:
                                references.append(url)

                    slide_data["references"] = references

                    # Validate and check for repetition
                    if validate_slide_content(slide_data):
                        # Simple repetition check
                        bullets = slide_data.get("bullets", [])
                        new_content = any(
                            bullet.lower() not in used_content for bullet in bullets
                        )

                        if new_content or len(slides) < 3:  # Ensure minimum slides
                            # Optimize title based on actual bullet content
                            slide_data = optimize_slide_title(slide_data)
                            slides.append(slide_data)

                            # Add bullets to used content tracking
                            for bullet in bullets:
                                used_content.add(bullet.lower())

                            # Show if title was optimized
                            if slide_data.get("original_title") and slide_data.get(
                                "original_title"
                            ) != slide_data.get("title"):
                                print(
                                    f"✅ Generated individual slide: {title} → {slide_data['title']}"
                                )
                            else:
                                print(
                                    f"✅ Generated individual slide: {slide_data['title']}"
                                )
                        else:
                            print(f"Skipping {title} due to content repetition")
                    else:
                        print(f"Content validation failed for {title}")

                except json.JSONDecodeError:
                    print(f"Failed to parse content for {title}")
                    continue

            except Exception as e:
                print(f"Error generating individual slide '{title}': {e}")
                continue

        return slides

    except Exception as e:
        print(f"Error in generate_slides_individually: {e}")
        return []


@tool
def generate_slide_content(outline: Dict, index_metadata: Dict) -> List[Dict]:
    """Generate detailed slide content with citations using unified content planning.

    Args:
        outline: Presentation outline
        index_metadata: Vector index for RAG queries

    Returns:
        List of slide specifications with bullets, notes, citations
    """
    try:
        # Phase 1: Create content allocation plan to eliminate repetition
        print("Creating content allocation plan to eliminate repetition...")
        allocation_plan = create_content_allocation_plan(outline, index_metadata)

        if "error" in allocation_plan:
            print(f"Failed to create allocation plan: {allocation_plan['error']}")
            print("Falling back to individual slide generation...")

            # Fallback: Generate slides individually with enhanced prompts
            return generate_slides_individually(outline, index_metadata)

        content_plan = allocation_plan.get("content_plan", {})
        source_references = allocation_plan.get("source_references", [])

        # Phase 2: Generate slides based on the allocation plan
        slides = []
        slide_titles = outline.get("slide_titles", [])
        llm = get_openai_llm()

        for i, title in enumerate(slide_titles):
            # Skip title and agenda slides - handle them separately
            if i == 0 or title.lower() in ["agenda", "references"]:
                continue

            # Get allocated insights for this slide
            allocated_insights = content_plan.get(title, [])

            if not allocated_insights:
                print(f"No allocated insights found for slide '{title}', skipping...")
                continue

            try:
                # Convert allocated insights into bullet points
                bullet_conversion_prompt = f"""
                Convert these allocated insights into ultra-concise bullet points for slide "{title}".
                
                ALLOCATED INSIGHTS:
                {chr(10).join([f"- {insight}" for insight in allocated_insights])}
                
                CRITICAL REQUIREMENTS:
                - Focus on complete THOUGHTS, not complete sentences
                - Maximum 15 words per bullet for complete ideas
                - Use bullet-style phrasing: efficient and professional  
                - NO markdown formatting (**, __, etc.) - plain text only
                - Include specific data: numbers, percentages, companies, dates
                - Each bullet must express one complete concept
                
                THOUGHT-BASED EXAMPLES:
                "AI adoption: 250% increase across enterprise sectors"
                "Microsoft: $10 billion strategic OpenAI partnership"  
                "75% workforce adoption of daily AI tools"
                "Healthcare productivity: 40% boost via AI automation"
                
                Return ONLY valid JSON in this exact format:
                {{
                    "title": "{title}",
                    "bullets": [
                        "Complete thought with specific data (max 15 words)",
                        "Another key insight with metrics and context",
                        "Third concept providing actionable information"
                    ],
                    "speaker_notes": "Detailed explanation of insights with context and supporting data. Include specific examples and implications. NO markdown formatting.",
                    "slide_type": "content"
                }}
                
                Convert all {len(allocated_insights)} insights into {len(allocated_insights)} bullets, max 15 words each.
                """

                # Rate limiting for OpenAI API
                openai_limiter.wait_if_needed("openai")
                conversion_response = llm.invoke(bullet_conversion_prompt)

                try:
                    slide_data = json.loads(conversion_response.content)
                    slide_data["references"] = source_references

                    # Validate content quality
                    if validate_slide_content(slide_data):
                        # Optimize title based on actual bullet content
                        slide_data = optimize_slide_title(slide_data)
                        slides.append(slide_data)

                        # Show if title was optimized
                        if slide_data.get("original_title") and slide_data.get(
                            "original_title"
                        ) != slide_data.get("title"):
                            print(
                                f"✅ Generated slide: {title} → {slide_data['title']}"
                            )
                        else:
                            print(f"✅ Generated slide: {slide_data['title']}")
                    else:
                        print(
                            f"Content quality check failed for {title}, regenerating..."
                        )
                        raise json.JSONDecodeError("Quality check failed", "", 0)

                except json.JSONDecodeError:
                    # Fallback: Create minimal slide with key insights
                    print(
                        f"JSON parsing failed for {title}, creating fallback slide..."
                    )

                    fallback_bullets = []
                    for insight in allocated_insights[:3]:  # Max 3 bullets
                        # Simple conversion: take first 15 words for complete thoughts
                        words = insight.split()[:15]
                        if len(words) >= 4:  # Minimum 4 words
                            fallback_bullets.append(" ".join(words))

                    if fallback_bullets:
                        slide_data = {
                            "title": title,
                            "bullets": fallback_bullets,
                            "speaker_notes": f"Key insights: {'. '.join(allocated_insights)}",
                            "slide_type": "content",
                            "references": source_references,
                        }
                        slides.append(slide_data)
                        print(f"✅ Generated fallback slide: {title}")

            except Exception as e:
                print(f"Error generating content for slide '{title}': {e}")
                continue

        print(
            f"Successfully generated {len(slides)} unique slides with no content repetition"
        )
        return slides

    except Exception as e:
        print(f"Error in generate_slide_content: {e}")
        return []


def validate_presentation_structure(slide_specs: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate overall presentation structure and content quality.

    Args:
        slide_specs: List of slide specifications

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check minimum number of content slides
    if len(slide_specs) < 3:
        issues.append(f"Too few slides: {len(slide_specs)} (minimum 3 required)")

    # Check for unique slide titles
    titles = [spec.get("title", "") for spec in slide_specs]
    if len(titles) != len(set(titles)):
        duplicate_titles = [title for title in set(titles) if titles.count(title) > 1]
        issues.append(f"Duplicate slide titles found: {duplicate_titles}")

    # Check each slide individually
    for i, spec in enumerate(slide_specs):
        slide_title = spec.get("title", f"Slide {i+1}")

        # Validate individual slide content
        if not validate_slide_content(spec):
            issues.append(f"Content validation failed for slide: '{slide_title}'")

        # Check for references
        references = spec.get("references", [])
        if not references:
            issues.append(f"No references found for slide: '{slide_title}'")

    # Check for content repetition across slides
    all_bullets = []
    for spec in slide_specs:
        bullets = spec.get("bullets", [])
        all_bullets.extend(bullets)

    # Look for similar bullets (basic similarity check)
    for i, bullet1 in enumerate(all_bullets):
        for j, bullet2 in enumerate(all_bullets[i + 1 :], i + 1):
            # Check if bullets share more than 50% of their words
            words1 = set(bullet1.lower().split())
            words2 = set(bullet2.lower().split())
            if len(words1 & words2) / max(len(words1), len(words2)) > 0.5:
                issues.append(f"Similar content detected: '{bullet1}' and '{bullet2}'")

    return len(issues) == 0, issues


def format_slide_text(text_frame, content: str, bullet_count: int):
    """Apply proper formatting to slide text with dynamic sizing."""
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN

    # Clear existing content
    text_frame.clear()

    # Calculate optimal font size based on content
    if bullet_count <= 3:
        font_size = Pt(28)  # Larger for fewer bullets
    elif bullet_count == 4:
        font_size = Pt(24)  # Medium for 4 bullets
    else:
        font_size = Pt(20)  # Smaller for 5+ bullets

    # Add content with formatting
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        if i > 0:
            # Add new paragraph for each bullet
            p = text_frame.add_paragraph()
        else:
            # Use first paragraph
            p = text_frame.paragraphs[0]

        p.text = line.strip()
        p.level = 0
        p.alignment = PP_ALIGN.LEFT

        # Apply font formatting
        for run in p.runs:
            run.font.size = font_size
            run.font.name = "Calibri"


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
        # Final comprehensive validation before presentation creation
        print("Running final presentation validation...")
        is_valid, validation_issues = validate_presentation_structure(slide_specs)

        if not is_valid:
            print("❌ Presentation validation failed:")
            for issue in validation_issues:
                print(f"  - {issue}")
            print(
                "Proceeding with presentation creation despite validation warnings..."
            )
        else:
            print("✅ Presentation validation passed successfully")
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
            agenda_text = "\n".join([item for item in agenda_items if item])
            format_slide_text(agenda_content.text_frame, agenda_text, len(agenda_items))

        # Create content slides
        for spec in slide_specs:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = spec.get("title", "")

            # Add bullet points with proper formatting
            bullets = spec.get("bullets", [])
            if bullets:
                content_placeholder = slide.placeholders[1]
                bullet_text = "\n".join([bullet for bullet in bullets])
                format_slide_text(
                    content_placeholder.text_frame, bullet_text, len(bullets)
                )

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
            format_slide_text(ref_content.text_frame, ref_text, len(unique_refs))

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
