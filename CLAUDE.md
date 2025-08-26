# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LLM-Powered PowerPoint Deck Builder

This project creates AI-powered research presentations by combining Brave Search API, LangChain, LlamaIndex, and python-pptx to generate fact-based PowerPoint decks with proper citations.

## Commands

### Development Environment Setup
```bash
# Create and activate virtual environment with uv
uv venv
uv sync

# Install dependencies from pyproject.toml
uv lock  # Create/update lockfile
uv sync --frozen  # Install from lockfile

# Run the application
uv run python deck_builder_cli.py

# Run with template support
uv run python deck_builder_cli.py --template TEMPLATE.pptx
```

### Testing & Quality
```bash
# Run tests with pytest
uv run pytest

# Code formatting with black
uv run black .

# Linting with ruff
uv run ruff check .
uv run ruff check --fix .  # Auto-fix issues
```

### Environment Variables
Create a `.env` file with:
```bash
BRAVE_API_KEY=brv-************************
OPENAI_API_KEY=sk-************************
```

## High-Level Architecture

### Core Pipeline Flow
1. **User Request** → Natural language prompt for deck topic
2. **Research Phase** → Brave Search API queries for recent, authoritative sources
3. **Content Loading** → LangChain WebBaseLoader fetches and parses pages
4. **Knowledge Indexing** → LlamaIndex creates vector index for grounded Q&A
5. **Outline Generation** → LLM creates JSON outline with slide structure
6. **Content Generation** → Per-slide JSON with bullets, speaker notes, citations
7. **Deck Creation** → python-pptx renders PPTX with optional template

### Key Components

**Research & Retrieval Stack:**
- `Brave Search API` - Web search for current information
- `LangChain WebBaseLoader` - Page fetching and parsing
- `LlamaIndex Vector Index` - RAG-based content grounding
- `BeautifulSoup4 + html2text` - HTML content extraction

**Generation Pipeline:**
- Two-phase LLM prompting (outline → detailed slides)
- Structured JSON output with strict schemas
- Citation tracking with inline references [1], [2]
- Dedicated References slide generation

**Presentation Layer:**
- `python-pptx` for PPTX generation
- Template support for corporate branding
- Slide types: Title, Agenda, Content, References

## Project-Specific Patterns

### Package Management with uv
- **NEVER use pip directly** - All dependencies managed through uv
- Use `uv sync` for reproducible environments
- Dependencies defined in `pyproject.toml` under `[project]` and `[tool.uv]`
- Lock dependencies with `uv lock` before deployment

### LLM Output Structure
Generate strict JSON for parsing:
```json
{
  "topic": "string",
  "objective": "string", 
  "slide_specs": [
    {
      "title": "string",
      "bullets": ["string"],
      "notes": "string with [citations]",
      "references": ["source urls"]
    }
  ],
  "references": ["all unique sources"]
}
```

### Citation Management
- Track sources throughout pipeline
- Use inline markers [1], [2] in speaker notes
- Deduplicate references before final slide
- Include publication dates when available

### Template Integration
When `template_path` is provided:
```python
prs = Presentation(template_path) if template_path else Presentation()
```
This inherits slide masters, themes, and corporate branding from the template.

## PydanticAI Development Workflow

### Core Principles
- **Start with INITIAL.md** - Define requirements in `PRPs/INITIAL.md` or `PRPs/INITIAL_llm_pptx_deck_builder.md`
- **Generate PRP** - `/generate-pydantic-ai-prp PRPs/INITIAL.md`
- **Execute PRP** - `/execute-pydantic-ai-prp PRPs/generated_prp.md`
- **Test with TestModel** - Validate logic before using real LLMs

### Agent Structure Pattern
```
src/
  ├── agent.py        # Main agent definition
  ├── tools.py        # Brave search, LlamaIndex tools
  ├── models.py       # Pydantic models for slides, references
  ├── dependencies.py # API keys, index management
  └── settings.py     # Environment configuration
```

### Environment Configuration
Follow `examples/main_agent_reference/settings.py` pattern:
```python
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    brave_api_key: str = Field(...)
    openai_api_key: str = Field(...)
    llm_model: str = Field(default="gpt-4")
```

## Archon Integration

### Task Management Workflow
1. Check current tasks: `archon:list_tasks(filter_by="status", filter_value="todo")`
2. Research before implementing: `archon:perform_rag_query()` + `archon:search_code_examples()`
3. Update task status: `archon:update_task(task_id="...", status="doing")`
4. Mark for review when complete: `archon:update_task(task_id="...", status="review")`

### Research Queries for This Project
- `archon:perform_rag_query(query="LangChain WebBaseLoader best practices", match_count=3)`
- `archon:perform_rag_query(query="LlamaIndex vector index optimization", match_count=3)`
- `archon:search_code_examples(query="python-pptx template usage", match_count=2)`
- `archon:search_code_examples(query="Brave Search API pagination", match_count=2)`

## Security Considerations

### API Key Management
- **NEVER commit API keys** - Use `.env` files
- Load with `python-dotenv` and `load_dotenv()`
- Validate keys on startup with pydantic validators
- Include `.env` in `.gitignore`

### Input Validation
- Sanitize user prompts before LLM calls
- Validate JSON schemas strictly
- Limit search result counts to prevent abuse

### API Rate Limiting & Best Practices
- **Rate Limiting**: Comprehensive rate limiting implemented for all API calls
  - Brave Search API: 0.5 req/sec, 30 req/min, 1800 req/hour
  - OpenAI API: 0.33 req/sec, 20 req/min, 1000 req/hour  
  - Web Scraping: 2 req/sec, 120 req/min, 7200 req/hour
- **Retry Logic**: Exponential backoff with jitter for failed requests
- **SSL Handling**: Robust SSL certificate handling for problematic sites
- **Timeout Management**: 30-second timeouts with retry strategies
- **User Agent**: Proper identification in HTTP headers

### Error Handling
- Never expose API keys in error messages
- Log errors without sensitive information
- Implement retry logic for transient failures
- Graceful degradation when services unavailable

## Common Development Tasks

### Adding New Slide Types
1. Extend slide generation JSON schema
2. Add rendering logic in `build_pptx()` function
3. Update agenda generation to include new type
4. Test with both default and template presentations

### Enhancing Research Quality
1. Adjust Brave Search parameters (freshness, count)
2. Tune LlamaIndex chunk size and overlap
3. Implement source quality scoring
4. Add domain-specific search filters

### Improving Citations
1. Extract publication dates from sources
2. Add author information when available
3. Implement Chicago/APA citation formats
4. Track citation usage per slide

## Testing Patterns

### Unit Testing
- Test JSON schema validation independently
- Mock Brave Search API responses
- Test citation extraction and deduplication
- Validate slide structure generation

### Integration Testing
- Test full pipeline with TestModel
- Verify template application
- Test with various prompt complexities
- Validate reference accuracy

## Future Enhancements to Consider

- **Chart Generation**: matplotlib/plotly integration for data visualization
- **Image Embedding**: Fetch and embed relevant images with attribution
- **Export Formats**: Google Slides, PDF, HTML presentations
- **Multi-language Support**: Detect and generate in multiple languages
- **Template Library**: Pre-built templates for common use cases
- **Streaming Generation**: Real-time slide preview during generation
- **Collaborative Features**: Multi-user deck building sessions