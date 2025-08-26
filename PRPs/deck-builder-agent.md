# LLM-Powered PowerPoint Deck Builder Agent

## 🎯 **AGENT OBJECTIVE**
Build a comprehensive LangGraph-powered agent that takes natural language requests for PowerPoint presentations, performs live web research via Brave Search API, creates grounded content using LlamaIndex RAG, and outputs polished `.pptx` files with proper citations and optional template support.

## 🏗 **AGENT ARCHITECTURE**

### **Agent Type**: LangGraph Multi-Node Workflow
- **Research Node**: Brave Search API + LangChain WebBaseLoader
- **Indexing Node**: LlamaIndex Vector Index creation
- **Content Generation Node**: Two-phase LLM prompting (outline → slides)
- **Presentation Node**: python-pptx rendering with template support

### **State Schema**
```python
from typing import List, Dict, Optional, TypedDict
from langchain_core.messages import AnyMessage

class DeckBuilderState(TypedDict):
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
```

## 📚 **CRITICAL CONTEXT**

### **LangChain Integration Patterns**
- **Documentation**: https://python.langchain.com/docs/concepts/tools/
- **Tool Creation**: Use `@tool` decorator with structured Pydantic inputs
- **State Management**: TypedDict with `Annotated[list, add_messages]` for message handling
- **LCEL Chains**: Use `|` operator for composing chains
- **Error Handling**: Implement retry logic with `RunnableLambda` wrappers

### **LlamaIndex RAG Patterns**
- **Documentation**: https://docs.llamaindex.ai/en/stable/examples/vector_stores/
- **Vector Store**: Use `VectorStoreIndex.from_documents(documents)`
- **Query Engine**: Create with `index.as_query_engine(similarity_top_k=5)`
- **Memory Management**: Use `ServiceContext` with chunk size optimization
- **Embeddings**: OpenAIEmbedding with proper API key management

### **Python-PPTX Integration**
- **Documentation**: https://python-pptx.readthedocs.io/en/latest/
- **Template Support**: `Presentation(template_path)` for branded decks
- **Slide Creation**: Use slide layouts and proper text formatting
- **Citations**: Implement inline reference markers [1], [2]

### **Brave Search API**
- **Documentation**: https://brave.com/search/api/
- **Authentication**: Bearer token in headers
- **Rate Limiting**: Implement backoff strategies
- **Result Parsing**: Extract URLs, titles, snippets for content loading

## 🛠 **IMPLEMENTATION BLUEPRINT**

### **Directory Structure**
```
src/
├── deck_builder_agent.py    # Main LangGraph workflow
├── tools.py                 # All @tool decorated functions
├── models.py                # Pydantic schemas
├── dependencies.py          # API keys, settings, LLM configs
├── settings.py              # Environment configuration
├── utils.py                 # Helper functions
└── templates/               # Default slide templates
```

### **Core Tools Implementation**

#### **1. Brave Search Tool**
```python
@tool
def search_web(query: str, count: int = 10) -> List[Dict]:
    """Search web using Brave Search API for current information.
    
    Args:
        query: Search terms
        count: Number of results (max 20)
        
    Returns:
        List of search results with URLs, titles, snippets
    """
    # Implementation with proper error handling and rate limiting
```

#### **2. Document Loading Tool**
```python
@tool
def load_web_documents(urls: List[str]) -> List[Dict]:
    """Load and parse web pages using LangChain WebBaseLoader.
    
    Args:
        urls: List of URLs to fetch and parse
        
    Returns:
        List of parsed Document objects with content and metadata
    """
    # Use WebBaseLoader with BeautifulSoup parsing
```

#### **3. Vector Index Creation Tool**
```python
@tool
def create_vector_index(documents: List[Dict]) -> Dict:
    """Create LlamaIndex vector store from documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Index metadata for query engine creation
    """
    # Use VectorStoreIndex.from_documents with OpenAI embeddings
```

#### **4. Content Generation Tools**
```python
@tool
def generate_outline(topic: str, index_metadata: Dict) -> Dict:
    """Generate presentation outline using RAG.
    
    Args:
        topic: Presentation topic
        index_metadata: Vector index metadata
        
    Returns:
        JSON outline with slide titles and flow
    """
    
@tool
def generate_slide_content(outline: Dict, index_metadata: Dict) -> List[Dict]:
    """Generate detailed slide content with citations.
    
    Args:
        outline: Presentation outline
        index_metadata: Vector index for RAG queries
        
    Returns:
        List of slide specifications with bullets, notes, citations
    """
```

#### **5. Presentation Generation Tool**
```python
@tool
def create_presentation(slide_specs: List[Dict], template_path: Optional[str] = None) -> str:
    """Generate PowerPoint file using python-pptx.
    
    Args:
        slide_specs: List of slide content dictionaries
        template_path: Optional PPTX template file path
        
    Returns:
        Path to generated PPTX file
    """
    # Use python-pptx with template support and proper formatting
```

### **LangGraph Workflow Implementation**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Define workflow
builder = StateGraph(DeckBuilderState)

# Add nodes
builder.add_node("research", research_node)
builder.add_node("load_docs", document_loading_node)  
builder.add_node("create_index", indexing_node)
builder.add_node("generate_outline", outline_node)
builder.add_node("generate_content", content_node)
builder.add_node("create_deck", presentation_node)

# Define flow
builder.add_edge(START, "research")
builder.add_edge("research", "load_docs")
builder.add_edge("load_docs", "create_index")
builder.add_edge("create_index", "generate_outline")
builder.add_edge("generate_outline", "generate_content")
builder.add_edge("generate_content", "create_deck")
builder.add_edge("create_deck", END)

# Compile with memory
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
```

## 🧪 **VALIDATION GATES**

All validation gates must be executable by the AI agent:

### **1. Syntax and Style Validation**
```bash
# Install and run code quality tools
uv add --dev ruff black mypy
uv run ruff check --fix src/
uv run black src/
uv run mypy src/
```

### **2. Unit Tests with FakeLLM**
```bash
# Test individual components
uv run python -m pytest tests/test_tools.py -v
uv run python -m pytest tests/test_models.py -v
uv run python -m pytest tests/test_agent.py -v
```

### **3. Integration Tests**
```bash
# Test full workflow with mocked APIs
uv run python -m pytest tests/test_integration.py -v
```

### **4. LangSmith Tracing Test**
```bash
# Verify observability
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-key>
uv run python tests/test_tracing.py
```

### **5. API Integration Test**
```bash
# Test with real APIs (requires keys)
uv run python tests/test_apis.py
```

## 📋 **IMPLEMENTATION TASKS**

### **Task 1: Project Setup and Dependencies** 
- Create `pyproject.toml` with uv dependencies
- Set up environment configuration with `settings.py`
- Initialize project structure with proper imports
- Configure API keys securely with `.env` support

### **Task 2: Core Tool Implementation**
- Implement Brave Search API tool with rate limiting
- Create LangChain WebBaseLoader integration
- Build LlamaIndex vector store creation tool
- Add error handling and retry mechanisms

### **Task 3: LangGraph Workflow Creation**
- Define `DeckBuilderState` TypedDict schema
- Implement each workflow node function
- Create state transitions and conditional logic
- Add memory persistence with checkpointer

### **Task 4: Content Generation System**
- Implement two-phase prompting (outline → slides)
- Create RAG queries for grounded content
- Build citation tracking system
- Add JSON schema validation

### **Task 5: Presentation Generation**
- Integrate python-pptx with template support
- Implement slide formatting and styling
- Create references slide generation
- Add proper error handling for file I/O

### **Task 6: Testing and Validation**
- Write unit tests with FakeLLM
- Create integration tests with mocked APIs
- Set up LangSmith tracing
- Add API integration tests

### **Task 7: Production Readiness**
- Add proper logging and monitoring
- Implement security best practices
- Create deployment configuration
- Add performance optimization

## ⚙ **ENVIRONMENT SETUP**

### **Dependencies (pyproject.toml)**
```toml
[project]
name = "llm-pptx-deck-builder"
version = "0.1.0"
description = "LangChain-powered PowerPoint deck builder with Brave Search and LlamaIndex RAG"
requires-python = ">=3.11"

dependencies = [
    "langchain>=0.2.16",
    "langchain-community>=0.2.16", 
    "langchain-openai>=0.2.1",
    "langgraph>=0.2.28",
    "llama-index>=0.11.16",
    "llama-index-llms-openai>=0.2.5",
    "llama-index-embeddings-openai>=0.2.3",
    "python-pptx>=0.6.23",
    "requests>=2.32.3",
    "beautifulsoup4>=4.12.3",
    "html2text>=2024.2.26",
    "tiktoken>=0.7.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.4.0",
    "python-dotenv>=1.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.3.2",
    "ruff>=0.6.9", 
    "black>=24.8.0",
    "mypy>=1.11.0",
    "langsmith>=0.1.0",
]
```

### **Environment Variables (.env)**
```bash
# API Keys
BRAVE_API_KEY=brv-************************
OPENAI_API_KEY=sk-************************

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__************************

# Optional: Model Configuration
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### **Development Commands**
```bash
# Environment setup
uv venv && uv sync

# Development workflow  
uv run python -m src.deck_builder_agent
uv run python -c "from src.deck_builder_agent import main; main()"

# Testing
uv run pytest
uv run ruff check --fix && uv run black . && uv run mypy src/

# Production
uv build
uv run python main.py --topic "AI in Education" --template corporate.pptx
```

## 🔒 **SECURITY CONSIDERATIONS**

### **API Key Management**
- Never commit API keys to version control
- Use environment variables with `.env` files
- Validate API keys on startup with pydantic validators
- Implement proper error handling without exposing keys

### **Input Validation** 
- Sanitize user prompts before LLM calls
- Validate JSON schemas strictly with Pydantic
- Limit search result counts to prevent abuse
- Implement rate limiting for API calls

### **Error Handling**
- Never expose API keys in error messages
- Log errors without sensitive information  
- Implement retry logic for transient failures
- Graceful degradation when services unavailable

## 🎖 **SUCCESS CRITERIA**

### **Functional Requirements**
- ✅ Accept natural language presentation requests
- ✅ Perform live web research with Brave Search
- ✅ Create grounded content using LlamaIndex RAG
- ✅ Generate structured JSON outline and slide content
- ✅ Output professional PPTX with citations
- ✅ Support optional PowerPoint templates

### **Technical Requirements**
- ✅ LangGraph workflow with proper state management
- ✅ Robust error handling and retry mechanisms
- ✅ Comprehensive test coverage (unit + integration)
- ✅ LangSmith observability integration
- ✅ Security best practices implementation
- ✅ Performance optimization for large documents

### **Quality Gates**
- ✅ All validation commands pass without errors
- ✅ Code coverage > 80% for core functions
- ✅ Documentation complete with examples
- ✅ API integration tests pass with real services
- ✅ Generated presentations are professional quality

## 📖 **REFERENCE DOCUMENTATION**

### **Core Technologies**
- **LangChain**: https://python.langchain.com/docs/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **LlamaIndex**: https://docs.llamaindex.ai/en/stable/
- **Python-PPTX**: https://python-pptx.readthedocs.io/en/latest/
- **Brave Search**: https://brave.com/search/api/

### **Code Examples**
- **LangGraph Multi-Agent**: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/
- **LangChain Tools**: https://python.langchain.com/docs/concepts/tools/
- **LlamaIndex RAG**: https://docs.llamaindex.ai/en/stable/examples/vector_stores/
- **Python-PPTX Examples**: https://github.com/scanny/python-pptx

### **Best Practices**
- **LangSmith Tracing**: https://docs.smith.langchain.com/
- **Security Guidelines**: https://python.langchain.com/docs/security/
- **Testing Patterns**: https://python.langchain.com/docs/concepts/tools/#testing
- **Production Deployment**: https://python.langchain.com/docs/guides/productionization/

---

**PRP Confidence Score: 9/10** - Comprehensive research-backed implementation plan with clear validation gates and real-world integration patterns.