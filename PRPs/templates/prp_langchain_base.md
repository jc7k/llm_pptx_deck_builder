---
name: "LangChain Agent PRP Template"
description: "Template for generating comprehensive PRPs for LangChain agent development projects"
---

## Purpose

[Brief description of the LangChain agent to be built and its main purpose]

## Core Principles

1. **LangChain Best Practices**: Deep integration with LangChain patterns for agent creation, chains, tools, and memory
2. **Production Ready**: Include security, observability with LangSmith, and monitoring for production deployments
3. **Type Safety**: Leverage Pydantic models for input/output validation throughout
4. **Composability**: Use LCEL (LangChain Expression Language) for building complex chains
5. **Comprehensive Testing**: Use FakeLLM and mock patterns for thorough agent validation

## ⚠️ Implementation Guidelines: Don't Over-Engineer

**IMPORTANT**: Keep your agent implementation focused and practical. Don't build unnecessary complexity.

### What NOT to do:
- ❌ **Don't create dozens of tools** - Build only the tools your agent actually needs
- ❌ **Don't over-complicate chains** - Keep LCEL chains simple and readable
- ❌ **Don't add unnecessary abstractions** - Follow LangChain patterns directly
- ❌ **Don't build complex state graphs** unless specifically required
- ❌ **Don't add custom parsers** unless standard output parsers won't work
- ❌ **Don't build in the examples/ folder**

### What TO do:
- ✅ **Start simple** - Build the minimum viable agent that meets requirements
- ✅ **Add tools incrementally** - Implement only what the agent needs to function
- ✅ **Follow LangChain docs** - Use proven patterns, don't reinvent
- ✅ **Use standard components** - ChatPromptTemplate, ChatOpenAI, create_react_agent
- ✅ **Test early and often** - Use FakeLLM to validate as you build

### Key Question:
**"Does this agent really need this feature to accomplish its core purpose?"**

If the answer is no, don't build it. Keep it simple, focused, and functional.

---

## Goal

[Detailed description of what the agent should accomplish]

## Why

[Explanation of why this agent is needed and what problem it solves]

## What

### Agent Type Classification
- [ ] **ReAct Agent**: Tool-using agent with reasoning and action capabilities
- [ ] **Conversational Agent**: Chat interface with memory and context
- [ ] **LCEL Chain**: Composable chain using LangChain Expression Language
- [ ] **Multi-Agent System**: Multiple agents with supervisor/orchestration
- [ ] **RAG Agent**: Retrieval-augmented generation with vector stores

### Model Provider Requirements
- [ ] **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- [ ] **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- [ ] **Google**: `gemini-pro`, `gemini-pro-vision`
- [ ] **Open Source**: `llama2`, `mistral`, via Ollama or HuggingFace
- [ ] **Fallback Strategy**: Multiple provider support with automatic failover

### External Integrations
- [ ] **Vector Stores**: Chroma, Pinecone, Weaviate, Qdrant, FAISS
- [ ] **LlamaIndex Integration**: VectorStoreIndex, SimpleDirectoryReader, RouterQueryEngine
- [ ] **Document Loaders**: PDF, CSV, JSON, Word, PowerPoint, web scraping
- [ ] **Embedding Models**: OpenAI, Cohere, HuggingFace, local embeddings
- [ ] **Web Search**: Brave, Tavily, SerpAPI, DuckDuckGo
- [ ] **Database Connections**: SQL, MongoDB, Neo4j, Redis
- [ ] **REST API Integrations**: List required external services

### Success Criteria
- [ ] Agent successfully handles specified use cases
- [ ] All tools work correctly with proper error handling
- [ ] Memory and state management functions as expected
- [ ] Comprehensive test coverage with FakeLLM and mocks
- [ ] Security measures implemented (API keys, input validation)
- [ ] LangSmith tracing enabled for observability
- [ ] Performance meets requirements (response time, token usage)

## All Needed Context

### LangChain Documentation & Research

```yaml
# MCP servers
- mcp: Archon
  query: "LangChain agent creation tools memory chains LCEL"
  why: Core framework understanding and latest patterns

# ESSENTIAL LANGCHAIN DOCUMENTATION - Must be researched
- url: https://python.langchain.com/docs/concepts/
  why: Core concepts including chat models, prompts, chains
  content: Architecture overview, component interfaces, best practices

- url: https://python.langchain.com/docs/concepts/agents/
  why: Agent creation patterns and tool integration
  content: ReAct pattern, tool calling, agent executors

- url: https://langchain-ai.github.io/langgraph/
  why: Stateful agent orchestration with LangGraph
  content: StateGraph, create_react_agent, multi-agent patterns

- url: https://python.langchain.com/docs/concepts/lcel/
  why: LangChain Expression Language for composable chains
  content: RunnablePassthrough, RunnableLambda, chain composition

- url: https://python.langchain.com/docs/concepts/memory/
  why: Memory and conversation management patterns
  content: ConversationBufferMemory, ConversationSummaryMemory, state persistence

# LLAMAINDEX INTEGRATION DOCUMENTATION - For RAG use cases
- url: https://docs.llamaindex.ai/en/stable/getting_started/concepts/
  why: Core LlamaIndex concepts for vector-based retrieval
  content: VectorStoreIndex, SimpleDirectoryReader, query engines, embeddings

- url: https://docs.llamaindex.ai/en/stable/module_guides/indexing/
  why: Document indexing and vector store patterns
  content: Document loading, text splitting, vector storage, index management

- url: https://docs.llamaindex.ai/en/stable/module_guides/querying/
  why: Query engine patterns and retrieval strategies
  content: RetrieverQueryEngine, RouterQueryEngine, SubQuestionQueryEngine

# Example patterns from research
- example: create_react_agent
  code: |
    from langgraph.prebuilt import create_react_agent
    
    agent = create_react_agent(
        model="openai:gpt-4",
        tools=[web_search, calculator],
        prompt="You are a research assistant..."
    )
    
- example: LCEL chain
  code: |
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    
    chain = (
        ChatPromptTemplate.from_template("Tell me about {topic}")
        | ChatOpenAI(model="gpt-4")
        | StrOutputParser()
    )

- example: LlamaIndex Vector RAG
  code: |
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    
    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()
    
    # Create vector index with embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4")
    
    index = VectorStoreIndex.from_documents(
        documents, 
        embed_model=embed_model
    )
    
    # Create retriever and query engine
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        llm=llm
    )
    
    # Query the index
    response = query_engine.query("What are the main topics?")

- example: LangChain + LlamaIndex Integration
  code: |
    from langchain.tools import tool
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.query_engine import RetrieverQueryEngine
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    
    # Setup LlamaIndex vector store
    documents = SimpleDirectoryReader("./knowledge_base").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    class KnowledgeSearchInput(BaseModel):
        query: str = Field(description="Search query for knowledge base")
    
    @tool("knowledge_search", args_schema=KnowledgeSearchInput)
    def knowledge_search(query: str) -> str:
        """Search the knowledge base for relevant information."""
        try:
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    # Create ReAct agent with knowledge search tool
    model = ChatOpenAI(model="gpt-4", temperature=0)
    agent = create_react_agent(
        model=model,
        tools=[knowledge_search],
        prompt="You are a knowledge assistant with access to a specialized knowledge base."
    )

- example: Multi-Document RAG with Router
  code: |
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    from llama_index.llms.openai import OpenAI
    
    # Load different document sets
    policy_docs = SimpleDirectoryReader("./policies").load_data()
    technical_docs = SimpleDirectoryReader("./technical").load_data()
    
    # Create separate indexes
    policy_index = VectorStoreIndex.from_documents(policy_docs)
    technical_index = VectorStoreIndex.from_documents(technical_docs)
    
    # Create query engines
    policy_engine = policy_index.as_query_engine()
    technical_engine = technical_index.as_query_engine()
    
    # Create router with query engine tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=policy_engine,
            metadata={"name": "policy_search", 
                     "description": "Search company policies and procedures"}
        ),
        QueryEngineTool(
            query_engine=technical_engine,
            metadata={"name": "technical_search",
                     "description": "Search technical documentation and guides"}
        )
    ]
    
    # Router automatically selects appropriate engine
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=OpenAI()),
        query_engine_tools=query_engine_tools
    )

- example: Python-PPTX Basic Presentation
  code: |
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    
    # Create new presentation
    prs = Presentation()
    
    # Add title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "AI-Generated Research Presentation"
    subtitle.text = "Powered by LangChain and LlamaIndex"
    
    # Add bullet point slide
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    
    title_shape = slide.shapes.title
    body_shape = slide.shapes.placeholders[1]
    title_shape.text = 'Key Findings'
    
    tf = body_shape.text_frame
    tf.text = 'First key finding from research'
    
    # Add additional bullet points
    p = tf.add_paragraph()
    p.text = 'Second finding with supporting data'
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = 'Third finding with analysis'
    p.level = 1
    
    prs.save('research_presentation.pptx')

- example: Python-PPTX Advanced Formatting
  code: |
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    
    prs = Presentation()
    
    # Add blank slide for custom content
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)
    
    # Add custom textbox with formatting
    left = Inches(1)
    top = Inches(1.5) 
    width = Inches(8)
    height = Inches(1)
    
    textbox = slide.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    tf.text = "Executive Summary"
    
    # Format text
    p = tf.paragraphs[0]
    p.font.name = 'Arial'
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x42, 0x24, 0xE9)
    p.alignment = PP_ALIGN.CENTER
    
    # Add content paragraph
    p = tf.add_paragraph()
    p.text = "Based on comprehensive research analysis, the following recommendations emerge:"
    p.font.size = Pt(14)
    p.level = 0
    
    prs.save('formatted_presentation.pptx')

- example: LangChain Tool for PowerPoint Generation
  code: |
    from langchain.tools import tool
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any
    import json
    
    class SlideContent(BaseModel):
        title: str = Field(description="Slide title")
        bullets: List[str] = Field(description="List of bullet points")
        notes: str = Field(default="", description="Speaker notes")
    
    class PresentationData(BaseModel):
        title: str = Field(description="Presentation title")
        subtitle: str = Field(description="Presentation subtitle") 
        slides: List[SlideContent] = Field(description="List of slide content")
    
    @tool("generate_powerpoint", args_schema=PresentationData)
    def generate_powerpoint(
        title: str,
        subtitle: str, 
        slides: List[Dict[str, Any]]
    ) -> str:
        """Generate a PowerPoint presentation from structured data."""
        try:
            prs = Presentation()
            
            # Add title slide
            title_slide_layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(title_slide_layout)
            slide.shapes.title.text = title
            slide.placeholders[1].text = subtitle
            
            # Add content slides
            for slide_data in slides:
                bullet_slide_layout = prs.slide_layouts[1]
                slide = prs.slides.add_slide(bullet_slide_layout)
                
                # Set title
                slide.shapes.title.text = slide_data.get('title', 'Untitled')
                
                # Add bullet points
                body_shape = slide.shapes.placeholders[1]
                tf = body_shape.text_frame
                
                bullets = slide_data.get('bullets', [])
                if bullets:
                    tf.text = bullets[0]
                    for bullet in bullets[1:]:
                        p = tf.add_paragraph()
                        p.text = bullet
                        p.level = 1
                
                # Add speaker notes
                notes = slide_data.get('notes', '')
                if notes:
                    slide.notes_slide.notes_text_frame.text = notes
            
            # Save presentation
            filename = f"{title.replace(' ', '_').lower()}.pptx"
            prs.save(filename)
            return f"Successfully created PowerPoint presentation: {filename}"
            
        except Exception as e:
            return f"Error creating PowerPoint: {str(e)}"

- example: Reading Existing PowerPoint Files
  code: |
    from pptx import Presentation
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    from typing import List, Dict
    
    class PowerPointExtractor(BaseModel):
        file_path: str = Field(description="Path to PowerPoint file")
    
    @tool("extract_pptx_content", args_schema=PowerPointExtractor)
    def extract_pptx_content(file_path: str) -> str:
        """Extract text content from existing PowerPoint presentation."""
        try:
            prs = Presentation(file_path)
            content = []
            
            for i, slide in enumerate(prs.slides):
                slide_content = {
                    'slide_number': i + 1,
                    'title': '',
                    'text_content': [],
                    'speaker_notes': ''
                }
                
                # Extract text from all shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        if shape == slide.shapes.title:
                            slide_content['title'] = shape.text
                        else:
                            slide_content['text_content'].append(shape.text)
                
                # Extract speaker notes
                if slide.has_notes_slide:
                    notes_slide = slide.notes_slide
                    if hasattr(notes_slide.notes_text_frame, 'text'):
                        slide_content['speaker_notes'] = notes_slide.notes_text_frame.text
                
                content.append(slide_content)
            
            return json.dumps(content, indent=2)
            
        except Exception as e:
            return f"Error reading PowerPoint file: {str(e)}"
```

### Agent Architecture Research

```yaml
# LangChain Architecture Patterns
agent_structure:
  configuration:
    - .env: Environment variables for API keys
    - config.py: Model and provider configuration
    - Use python-dotenv for loading environment variables
    - Never hardcode API keys or model names
  
  agent_definition:
    - Use create_react_agent for tool-using agents
    - Use StateGraph for complex workflows
    - ChatPromptTemplate for prompt engineering
    - Proper output parsers (StrOutputParser, PydanticOutputParser)
  
  tool_integration:
    - @tool decorator for custom tools
    - Structured tool inputs with Pydantic
    - Tool error handling with try/except
    - Return clear, actionable tool outputs
  
  memory_management:
    - ConversationBufferMemory for short conversations
    - ConversationSummaryMemory for long conversations
    - Custom memory with BaseMemory interface
    - Checkpointing for state persistence
  
  testing_strategy:
    - FakeLLM for deterministic testing
    - Mock tool responses
    - Integration testing with real models
    - LangSmith for production monitoring
```

### Security and Production Considerations

```yaml
# LangChain Security Patterns (research required)
security_requirements:
  api_management:
    environment_variables: ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGCHAIN_API_KEY"]
    secure_storage: "Use python-dotenv, never commit .env files"
    rotation_strategy: "Plan for key rotation and management"
  
  input_validation:
    sanitization: "Validate all user inputs before processing"
    prompt_injection: "Use input guards and content filtering"
    rate_limiting: "Implement token and request limits"
  
  observability:
    langsmith_tracing: "Enable LANGCHAIN_TRACING_V2 for monitoring"
    custom_callbacks: "Implement callbacks for logging and metrics"
    error_tracking: "Structured error logging without exposing secrets"
```

### Common LangChain Gotchas (research and document)

```yaml
# Agent-specific gotchas to research and address
implementation_gotchas:
  streaming_complexity:
    issue: "Streaming responses with tools can be complex"
    research: "LangChain streaming patterns and callbacks"
    solution: "[To be documented based on research]"
  
  token_limits:
    issue: "Different models have different context windows"
    research: "Model comparison and token counting"
    solution: "[To be documented based on research]"
  
  memory_persistence:
    issue: "Memory doesn't persist across sessions by default"
    research: "Checkpointing and state persistence patterns"
    solution: "[To be documented based on research]"
  
  tool_error_handling:
    issue: "Tool failures can crash agent execution"
    research: "Error handling and retry patterns for tools"
    solution: "[To be documented based on research]"
```

## Implementation Blueprint

### Technology Research Phase

**RESEARCH REQUIRED - Complete before implementation:**

✅ **LangChain Framework Deep Dive:**
- [ ] Agent creation patterns with create_react_agent
- [ ] LCEL chain composition and best practices
- [ ] Tool integration patterns with @tool decorator
- [ ] Memory management and conversation persistence
- [ ] Testing strategies with FakeLLM and mocks

✅ **Agent Architecture Investigation:**
- [ ] Project structure conventions (agents/, chains/, tools/, memory/)
- [ ] Prompt template design with ChatPromptTemplate
- [ ] Output parsing with structured parsers
- [ ] Streaming patterns and callbacks
- [ ] Error handling and retry mechanisms

✅ **Security and Production Patterns:**
- [ ] API key management with python-dotenv
- [ ] Input validation and sanitization
- [ ] LangSmith integration for observability
- [ ] Rate limiting and cost management
- [ ] Deployment and scaling considerations

### Agent Implementation Plan

```yaml
Implementation Task 1 - Project Setup and Configuration:
  CREATE project structure:
    - agents/: Agent definitions and orchestration
    - chains/: LCEL chain definitions
    - tools/: Custom tool implementations
    - memory/: Memory and state management
    - prompts/: Prompt templates
    - tests/: Comprehensive test suite
    - .env.example: Environment variable template

Implementation Task 2 - Core Agent Development:
  IMPLEMENT agent using LangChain patterns:
    - Use create_react_agent for tool-using agents
    - ChatPromptTemplate for prompt engineering
    - Environment-based model configuration
    - Proper error handling and logging
    - Example:
      ```python
      from langgraph.prebuilt import create_react_agent
      from langchain_openai import ChatOpenAI
      
      model = ChatOpenAI(model="gpt-4", temperature=0)
      agent = create_react_agent(
          model=model,
          tools=tools,
          prompt=agent_prompt
      )
      ```

Implementation Task 3 - Tool Integration:
  DEVELOP custom tools:
    - Tool functions with @tool decorator
    - Structured inputs with Pydantic models
    - Clear tool descriptions for LLM understanding
    - Error handling and retry logic
    - Example:
      ```python
      from langchain.tools import tool
      from pydantic import BaseModel, Field
      
      class SearchInput(BaseModel):
          query: str = Field(description="Search query")
      
      @tool("web_search", args_schema=SearchInput)
      def web_search(query: str) -> str:
          """Search the web for information."""
          # Implementation here
          return results
      ```

Implementation Task 4 - Vector Store and RAG Integration:
  IMPLEMENT LlamaIndex vector retrieval:
    - Document loading with SimpleDirectoryReader
    - Vector index creation with VectorStoreIndex
    - Query engine setup with proper embedding models
    - Integration with LangChain tools via @tool decorator
    - Example:
      ```python
      from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
      from langchain.tools import tool
      
      # Load and index documents
      documents = SimpleDirectoryReader("./docs").load_data()
      index = VectorStoreIndex.from_documents(documents)
      query_engine = index.as_query_engine()
      
      @tool("search_documents")
      def search_docs(query: str) -> str:
          response = query_engine.query(query)
          return str(response)
      ```

Implementation Task 5 - Memory and State Management:
  IMPLEMENT memory patterns:
    - ConversationBufferMemory for simple use cases
    - ConversationSummaryMemory for long conversations
    - Custom memory implementations as needed
    - State persistence with checkpointing

Implementation Task 6 - Comprehensive Testing:
  CREATE test suite:
    - FakeLLM for deterministic unit tests
    - Mock tool responses for integration tests
    - End-to-end testing with test models
    - Performance and token usage testing
    - Vector store and RAG testing
    - Example:
      ```python
      from langchain.llms.fake import FakeLLM
      
      fake_llm = FakeLLM(responses=["test response"])
      test_agent = create_react_agent(
          model=fake_llm,
          tools=mock_tools
      )
      ```

Implementation Task 7 - Production Configuration:
  SETUP production patterns:
    - Environment variable management
    - LangSmith tracing configuration
    - Error handling and monitoring
    - Rate limiting and cost tracking
    - Vector store persistence and scaling
    - Deployment configuration
```

## Validation Loop

### Level 1: Project Structure Validation

```bash
# Verify complete LangChain project structure
find . -type f -name "*.py" | grep -E "(agents|chains|tools|memory|prompts)" | head -20
test -f .env.example && echo "Environment template present"
test -d tests && echo "Test directory present"

# Verify proper LangChain imports
grep -r "from langchain" . --include="*.py" | head -5
grep -r "from langgraph" . --include="*.py" | head -5
grep -r "@tool" . --include="*.py" | head -5

# Expected: All required directories with proper LangChain patterns
# If missing: Generate missing components with correct patterns
```

### Level 2: Agent Functionality Validation

```bash
# Test agent can be imported and instantiated
python -c "
from agents.main import agent
print('Agent created successfully')
print(f'Tools: {len(agent.tools)}')
"

# Test with FakeLLM for validation
python -c "
from langchain.llms.fake import FakeLLM
from agents.main import create_agent

fake_llm = FakeLLM(responses=['test response'])
test_agent = create_agent(llm=fake_llm)
result = test_agent.invoke({'input': 'test query'})
print(f'Agent response: {result}')
"

# Expected: Agent instantiation works, tools registered, FakeLLM validation passes
# If failing: Debug agent configuration and tool registration
```

### Level 3: Comprehensive Testing Validation

```bash
# Run complete test suite
python -m pytest tests/ -v

# Test specific agent behavior
python -m pytest tests/test_agent.py::test_agent_response -v
python -m pytest tests/test_tools.py::test_tool_validation -v
python -m pytest tests/test_memory.py::test_conversation_memory -v

# Test LangSmith integration
LANGCHAIN_TRACING_V2=true python tests/test_tracing.py

# Expected: All tests pass, comprehensive coverage achieved
# If failing: Fix implementation based on test failures
```

### Level 4: Production Readiness Validation

```bash
# Verify security patterns
grep -r "API_KEY" . | grep -v ".env" | grep -v ".py:" # Should not expose keys
test -f .env.example && echo "Environment template present"

# Check error handling
grep -r "try:" . --include="*.py" | wc -l  # Should have error handling
grep -r "except" . --include="*.py" | wc -l  # Should have exception handling

# Verify LangSmith setup
grep -r "LANGCHAIN_TRACING_V2" . --include="*.py" | wc -l  # Should have tracing

# Expected: Security measures in place, error handling comprehensive, tracing configured
# If issues: Implement missing security and production patterns
```

## Final Validation Checklist

### Agent Implementation Completeness

- [ ] Complete LangChain project structure: `agents/`, `chains/`, `tools/`, `memory/`, `prompts/`
- [ ] Agent instantiation with proper model configuration
- [ ] Tool registration with @tool decorator and descriptions
- [ ] Memory implementation with appropriate strategy
- [ ] LCEL chains properly composed and tested
- [ ] Comprehensive test suite with FakeLLM and mocks

### LangChain Best Practices

- [ ] Type safety with Pydantic models for tool inputs
- [ ] Security patterns implemented (API keys, input validation)
- [ ] Error handling and retry mechanisms for robust operation
- [ ] Streaming patterns appropriate for use case
- [ ] Documentation and code comments for maintainability

### Production Readiness

- [ ] Environment configuration with .env files
- [ ] LangSmith tracing enabled for observability
- [ ] Performance optimization and token management
- [ ] Deployment readiness with proper configuration
- [ ] Maintenance and update strategies documented

---

## Anti-Patterns to Avoid

### LangChain Agent Development

- ❌ Don't skip FakeLLM testing - always test with FakeLLM during development
- ❌ Don't hardcode API keys - use environment variables for all credentials
- ❌ Don't ignore streaming complexities - plan for streaming from the start
- ❌ Don't create overly complex chains - keep LCEL chains readable
- ❌ Don't skip error handling - implement comprehensive retry and fallback mechanisms

### Agent Architecture

- ❌ Don't mix agent patterns - clearly separate ReAct, conversational, and chain patterns
- ❌ Don't ignore memory limits - plan for conversation truncation and summarization
- ❌ Don't skip output validation - always validate and parse LLM outputs
- ❌ Don't forget tool documentation - ensure all tools have clear descriptions

### Security and Production

- ❌ Don't expose sensitive data - validate all outputs and logs for security
- ❌ Don't skip input validation - sanitize and validate all user inputs
- ❌ Don't ignore rate limiting - implement proper throttling for API calls
- ❌ Don't deploy without monitoring - include LangSmith tracing from the start

**RESEARCH STATUS: [TO BE COMPLETED]** - Complete comprehensive LangChain research before implementation begins.