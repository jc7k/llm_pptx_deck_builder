# LLM-Powered PowerPoint Deck Builder

AI-powered research presentation generator that creates professional PowerPoint decks with factual content and proper citations. Combines Brave Search API, LlamaIndex RAG, and python-pptx to generate high-quality, research-backed presentations.

## Features

- **Research-Driven Content**: Uses Brave Search API to gather current, authoritative sources
- **RAG-Enhanced Generation**: LlamaIndex vector indexing for grounded, factual content
- **Professional Quality**: Complete thought validation, dynamic formatting, and citation tracking
- **Template Support**: Works with custom PowerPoint templates for branding consistency
- **Verbose Progress Tracking**: Real-time updates during the multi-minute generation process

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jc7k/llm_pptx_deck_builder.git
cd llm_pptx_deck_builder

# Create and activate virtual environment with uv
uv venv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# BRAVE_API_KEY=brv-************************
# OPENAI_API_KEY=sk-************************
```

### Basic Usage

```bash
# Generate a presentation
uv run python deck_builder_cli.py --topic "AI impact on job market trends 2025"

# Use a custom template
uv run python deck_builder_cli.py --topic "Market Analysis" --template corporate_template.pptx
```

## Example Output

Here's what you'll see when generating a presentation:

```bash
$ python deck_builder_cli.py --topic "AI impact on job market trends 2025"
🚀 Starting presentation generation for: AI impact on job market trends 2025
This may take several minutes...
[11:01:38] 🔍 RESEARCH: Starting web search for: AI impact on job market trends 2025
[11:01:38] 🔍 RESEARCH: Applying rate limiting for Brave Search API...
[11:01:38] 🔍 RESEARCH: Enhanced search query: AI impact on job market trends 2025 statistics data trends 2025 report analysis...
[11:01:38] 🔍 RESEARCH: Querying Brave Search API...
[11:01:41] 🔍 RESEARCH: ✅ Found 15 search results
[11:01:41] 🔍 RESEARCH: Top sources: https://www.nexford.edu/insights/how-will-ai-affect-jobs, https://thehill.com/policy/technology/5460357-ai-impact-on-job-market/, https://www.gartner.com/en/articles/hype-cycle-for-artificial-intelligence, https://id-times.com/finance/2025-job-market/, https://www.stlouisfed.org/on-the-economy/2025/aug/recent-college-grads-bear-brunt-labor-market-shifts
[11:01:41] 📄 DOCUMENTS: Processing 15 search results...
[11:01:41] 📄 DOCUMENTS: Extracting URLs from search results...
[11:01:41] 📄 DOCUMENTS: Found 15 URLs to process
[11:01:41] 📄 DOCUMENTS: Starting document loading with rate limiting...
[11:01:55] 📄 DOCUMENTS: ✅ Successfully loaded 14 documents
[11:01:55] 📄 DOCUMENTS: Total content: 226,564 characters
[11:01:55] 🧠 INDEXING: Creating vector index from 14 documents...
[11:01:55] 🧠 INDEXING: Converting documents to LlamaIndex format...
[11:01:55] 🧠 INDEXING: Building vector embeddings with OpenAI...
[11:01:57] 🧠 INDEXING: ✅ Vector index created: 14 docs, 66 chunks
[11:01:57] 🧠 INDEXING: Index ID: deck_builder_1756231317
[11:01:57] 📋 OUTLINE: Generating presentation outline for: AI impact on job market trends 2025
[11:01:57] 📋 OUTLINE: Querying vector index for relevant content...
[11:01:57] 📋 OUTLINE: Applying rate limiting for OpenAI API...
[11:02:21] 📋 OUTLINE: ✅ Generated outline with 11 slides
[11:02:21] 📋 OUTLINE: Slides: Title Slide, Agenda, Introduction, Key Concepts, Current Trends...
[11:02:21] 📋 OUTLINE: Estimated duration: 15 minutes
[11:02:21] 📝 CONTENT: Generating detailed content for 11 slides...
[11:02:21] 📝 CONTENT: Querying vector index for slide-specific content...
[11:02:21] 📝 CONTENT: Applying rate limiting for OpenAI API...
[11:02:21] 📝 CONTENT: Processing slides with RAG-based content generation...
Creating content allocation plan to eliminate repetition...
✅ Generated slide: Introduction → AI Market Growth Projections to 2025
✅ Generated slide: Key Concepts → Surge in AI Adoption Amid Labor Shortages
✅ Generated slide: Current Trends → Global AI Adoption in Healthcare and Workplaces
✅ Generated slide: Applications → AI's Economic Impact: Jobs and Revenue Growth
✅ Generated slide: Challenges → Global Workforce Impact of AI Distrust
✅ Generated slide: Future Outlook → AI's Economic Impact and Job Growth by 2030
✅ Generated slide: Conclusions → AI Market Growth and Workforce Concerns
✅ Generated slide: Next Steps → Building Trust and Skills for AI Success
Successfully generated 8 unique slides with no content repetition
[11:03:48] 📝 CONTENT: ✅ Generated content for 8 slides
[11:03:48] 📝 CONTENT: Collecting and processing citations...
[11:03:48] 📝 CONTENT: Slide 1: 3 citations
[11:03:48] 📝 CONTENT: Slide 2: 3 citations
[11:03:48] 📝 CONTENT: Slide 3: 3 citations
[11:03:48] 📝 CONTENT: Slide 4: 3 citations
[11:03:48] 📝 CONTENT: Slide 5: 3 citations
[11:03:48] 📝 CONTENT: Slide 6: 3 citations
[11:03:48] 📝 CONTENT: Slide 7: 3 citations
[11:03:48] 📝 CONTENT: Slide 8: 3 citations
[11:03:48] 📝 CONTENT: Total citations before deduplication: 24
[11:03:48] 📝 CONTENT: Deduplicating citations...
[11:03:48] 📝 CONTENT: ✅ Final unique citations: 3
[11:03:48] 📝 CONTENT: Generated 24 total bullet points across all slides
[11:03:48] 🎨 PRESENTATION: Creating PowerPoint file with 8 slides...
[11:03:48] 🎨 PRESENTATION: Using default PowerPoint template
[11:03:48] 🎨 PRESENTATION: Including 3 references
[11:03:48] 🎨 PRESENTATION: Initializing python-pptx presentation...
[11:03:48] 🎨 PRESENTATION: Rendering title slide...
[11:03:48] 🎨 PRESENTATION: Processing content slides...
Running final presentation validation...
❌ Presentation validation failed:
  - Similar content detected: 'Global AI market projected between $244 billion and $757.6 billion by 2025' and 'AI market projected to reach $190 billion by 2025'
  - Similar content detected: 'AI to add over $15 trillion to global revenue by 2030' and 'AI expected to contribute over $15 trillion to global revenue by 2030'
  - Similar content detected: 'AI market projected to reach $2.4 trillion by 2032' and 'AI market projected to reach $190 billion by 2025'
Proceeding with presentation creation despite validation warnings...
[11:03:48] 🎨 PRESENTATION: ✅ Presentation saved to: output/AI Market Growth Projections to 2025_20250826_110348.pptx
[11:03:48] 🎨 PRESENTATION: File size: 0.0 MB
[11:03:48] 🎨 PRESENTATION: 🎉 Presentation generation complete!
✅ Presentation generated successfully!
📄 Output file: output/AI Market Growth Projections to 2025_20250826_110348.pptx
🎯 Slides created: 8
📚 References included: 3
```

## Architecture

### Core Pipeline

1. **Research Phase**: Brave Search API gathers current, authoritative sources
2. **Document Loading**: LangChain WebBaseLoader fetches and parses web content  
3. **Knowledge Indexing**: LlamaIndex creates vector embeddings for RAG
4. **Outline Generation**: LLM creates structured slide outline from research
5. **Content Generation**: Per-slide content with bullet points, notes, and citations
6. **Quality Validation**: Complete thought validation and formatting checks
7. **Presentation Creation**: python-pptx renders professional PowerPoint output

### Key Components

- **Brave Search API**: Web search for current information with rate limiting
- **LangChain WebBaseLoader**: Page fetching with SSL handling and retry logic
- **LlamaIndex Vector Store**: RAG-based content grounding and retrieval
- **Content Quality System**: Validates complete thoughts, prevents repetition
- **Dynamic Formatting**: Auto-adjusts font sizes and layouts for readability
- **Citation Management**: Tracks sources with inline references and bibliography

## Configuration

### Environment Variables

```bash
# Required API Keys
BRAVE_API_KEY=brv-************************  # Brave Search API
OPENAI_API_KEY=sk-************************   # OpenAI GPT models

# Optional Settings
OPENAI_MODEL=gpt-4o                         # Default: gpt-4o  
USER_AGENT=llm-pptx-deck-builder/1.0        # For web scraping
```

### Rate Limiting

Production-ready rate limiting is implemented for all APIs:

- **Brave Search**: 0.5 req/sec, 30 req/min, 1800 req/hour
- **OpenAI API**: 0.33 req/sec, 20 req/min, 1000 req/hour  
- **Web Scraping**: 2 req/sec, 120 req/min, 7200 req/hour

## Content Quality Features

### Complete Thoughts Validation
- Ensures bullet points express complete thoughts (not necessarily complete sentences)
- Detects incomplete endings with prepositions, conjunctions, and articles
- Validates word count (4-15 words for optimal readability)
- Prevents markdown formatting artifacts

### Anti-Repetition System
- Content allocation planning prevents duplicate insights across slides
- Semantic similarity detection with automated retry mechanisms
- Specialized prompts for different slide types (Introduction, Applications, etc.)

### Title Optimization
- Automatic title-content alignment for semantic harmony
- Replaces generic titles with specific, insight-driven alternatives
- Length management prevents title line wraps (8-word maximum)
- Quote removal and formatting cleanup

## Development

### Testing

```bash
# Run all tests
uv run pytest

# Test content quality validation
uv run python test_final_validation.py

# Test thought completeness detection
uv run python test_validation_only.py

# Code quality checks
uv run ruff check .
uv run ruff check --fix .
```

### Project Structure

```
src/
├── deck_builder_agent.py    # Main LangGraph workflow
├── tools.py                 # Core generation and validation logic  
├── models.py                # Pydantic data models
├── dependencies.py          # LlamaIndex and API setup
├── settings.py              # Environment configuration
└── rate_limiter.py          # API rate limiting utilities

tests/
├── test_agent.py           # Agent workflow tests
├── test_tools.py           # Tool function tests
└── test_integration.py     # End-to-end tests
```

## Advanced Usage

### Custom Templates

Use corporate PowerPoint templates for branded presentations:

```bash
python deck_builder_cli.py \
  --topic "Q4 Financial Results" \
  --template templates/corporate_theme.pptx
```

### API Integration

The system can be integrated into other applications:

```python
from src.deck_builder_agent import DeckBuilderAgent
from src.dependencies import get_dependencies

# Initialize agent
deps = get_dependencies()
agent = DeckBuilderAgent(deps)

# Generate presentation
result = agent.generate_deck("Market Analysis 2025")
print(f"Generated: {result['output_file']}")
```

## Requirements

- Python 3.11+
- uv package manager
- Brave Search API key
- OpenAI API key
- Internet connection for research and embedding generation

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/jc7k/llm_pptx_deck_builder/issues) page.