# LLM-Powered Research Deck Builder (LangChain + LlamaIndex + Brave → PowerPoint)

**FEATURE:**  
An AI-assisted pipeline that takes a natural-language request for a PowerPoint deck, performs **live research via Brave Search API**, grounds content with **LlamaIndex retrieval**, and outputs a polished **`.pptx`** using `python-pptx`. The system emphasizes factual accuracy (citations per slide), clean structure (title → agenda → scoped sections → next steps), and a dedicated **References** slide. It also supports applying an **optional PowerPoint template** for corporate branding.

---

**EXAMPLES:**  
- **Executive Briefing:** “Latest advances in AI tutoring for K–12” → searches Brave, ingests top sources, produces a 12–15 slide briefing with concise bullets, rich speaker notes, and linked references.  
- **Policy Update:** “Implications of the EU AI Act for edtech vendors” → extracts key provisions, timelines, compliance guidance, and market impacts with citations.  
- **Competitive Landscape:** “SecureIQ Lab vs. Gartner & SE Labs” → builds a side-by-side comparison, methodology caveats, and recommended next steps.

---

**DOCUMENTATION:**  
- **Brave Search API:** https://brave.com/search/api/  
- **LangChain:** https://python.langchain.com/  
- **LlamaIndex:** https://docs.llamaindex.ai/  
- **python-pptx:** https://python-pptx.readthedocs.io/en/latest/  
- **OpenAI API:** https://platform.openai.com/docs/  

---

**TECH STACK:**  

- **Retrieval & Research**  
  - **Brave Search API** for recent, relevant links.  
  - **LangChain WebBaseLoader** to fetch & parse pages.  
  - **LlamaIndex Vector Index** for grounded Q&A over fetched content.

- **LLM Orchestration**  
  - Two-phase prompting:  
    1) **Outline JSON** (titles, purposes, flow).  
    2) **Per-slide JSON** (title, 3–5 bullets, 80–150 word speaker notes, citations).  
  - Output uses inline reference markers \[1], \[2] in notes and adds a References slide.

- **Presentation Generation**  
  - **`python-pptx`** to render Title, Agenda, Content, and References slides.  
  - **Optional Template**: user may provide a `.pptx` template. If given, the generated deck adopts its slide master, theme, and branding.

- **Environment & Dependencies**  
  - **`uv`** for virtualenv + dependency management (with **`pyproject.toml`**).  
  - Reproducible lockfile with **`uv lock`**.  
  - **No `pip` usage**.

---

**SYSTEM PROMPT PRINCIPLES (Summarized):**  
- Be precise, current, and well-cited; avoid hallucinations.  
- 3–5 scannable bullets per slide; notes include nuance and citations.  
- Prefer recent, authoritative sources; acknowledge uncertainty when evidence is weak.  
- Return **strict JSON** for machine-safe parsing.

---

## Minimal UV Setup

**`pyproject.toml`** (example)
```toml
[project]
name = "llm-research-deck-builder"
version = "0.1.0"
description = "Generate research-grounded PowerPoint decks using Brave Search + LangChain + LlamaIndex + python-pptx."
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
  "requests>=2.32.3",
  "python-pptx>=0.6.23",
  "langchain>=0.2.16",
  "langchain-community>=0.2.16",
  "llama-index>=0.11.16",
  "llama-index-llms-openai>=0.2.5",
  "llama-index-embeddings-openai>=0.2.3",
  "beautifulsoup4>=4.12.3",
  "html2text>=2024.2.26",
  "tiktoken>=0.7.0",
]

[tool.uv]
dev-dependencies = [
  "pytest>=8.3.2",
  "ruff>=0.6.9",
  "black>=24.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Create & sync environment (no pip):**
```bash
# Create a local virtual environment:
uv venv

# Install deps from pyproject (and lock):
uv sync

# (Optional) Update lockfile when you change deps:
uv lock

# Run the app:
uv run python main.py
```

**Environment variables:**
```bash
export BRAVE_API_KEY=brv-************************
export OPENAI_API_KEY=sk-************************
```

---

## Minimal Orchestration with Template Support

**`main.py`** (excerpt)
```python
from pptx import Presentation

def build_pptx(plan, output_path: str = "deck.pptx", template_path: Optional[str] = None):
    prs = Presentation(template_path) if template_path else Presentation()

    today = datetime.date.today().strftime("%b %d, %Y")
    add_title(prs, plan.topic, f"{plan.objective} · {today}")

    # Agenda
    agenda = [s.title for s in plan.slide_specs if s.title][:6]
    add_bullets(prs, "Agenda", agenda, "Overview of sections and flow.")

    # Content
    for s in plan.slide_specs:
        if s.title.lower() in {"agenda", plan.topic.lower()}:
            continue
        add_bullets(prs, s.title, s.bullets, s.notes)

    # References
    add_references(prs, dedupe(plan.references))

    prs.save(output_path)
    return output_path
```

Usage example with template:
```bash
uv run python main.py --template TEMPLATE.pptx
```

---

## Minimal Dockerfile (uv-native, with template support)

**`Dockerfile`**
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml ./
RUN uv venv && uv sync --frozen

COPY . .

# Optional: copy a template into the container
# COPY TEMPLATE.pptx ./

CMD ["uv", "run", "python", "main.py"]
```

---

## Directory Structure

```
/app
  ├─ main.py                 # Orchestration: search → index → outline → slides → pptx
  ├─ pyproject.toml          # Dependencies & metadata (managed by uv)
  ├─ Dockerfile              # uv-only container (no pip)
  ├─ TEMPLATE.pptx           # (optional) corporate PPT template for branding
```

---

**OTHER CONSIDERATIONS:**  
- **Target Users:** Executives, educators, consultants, analysts, sales engineers.  
- **Core V1:** Prompt → Brave search → page load → LlamaIndex RAG → JSON outline & slides → PPTX with references.  
- **Branding:** Optional `TEMPLATE.pptx` ensures generated deck inherits corporate theme.  
- **Future Enhancements:**  
  - Chart generation (e.g., `matplotlib`) and image embedding.  
  - Export to Google Slides.  
  - Multi-language support.  
  - On-the-fly template selection per request.

---
