# Repository Guidelines

## Project Structure & Module Organization
- src/: Core code (e.g., `deck_builder_agent.py`, `tools.py`, `settings.py`, `dependencies.py`, `rate_limiter.py`, `models.py`).
- deck_builder_cli.py: CLI entry for generating decks.
- tests/: Pytest suite (`test_*.py`) covering tools and workflow.
- examples/: Sample topics/templates; output/: generated `.pptx` files.
- .env.example → .env: API keys and optional config; pyproject.toml: deps + tooling.

## Build, Test, and Development Commands
- Create env: `uv venv && uv sync` (Python 3.11+).
- Run CLI: `uv run python deck_builder_cli.py --topic "AI in 2025" --verbose`.
- Validate config only: `uv run python deck_builder_cli.py --validate-only`.
- Run tests: `uv run pytest -v` (e.g., `-k tools`, `-m integration`).
- Lint: `uv run ruff check src tests`.
- Format: `uv run black .`.
- Type-check: `uv run mypy src`.

## Coding Style & Naming Conventions
- Style: Black (88 cols), Ruff for linting, type hints encouraged; MyPy clean for `src/`.
- Indentation: 4 spaces; imports grouped stdlib/third-party/local.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_CASE`.
- Docstrings: concise triple-quoted summaries; avoid overly long comments.

## Testing Guidelines
- Framework: Pytest. Structure tests in `tests/` as `test_*.py`, classes `Test*`, functions `test_*` (see `pytest.ini`).
- Mock external calls (OpenAI, Brave, network, PPTX) via `unittest.mock.patch` and fixtures (see `tests/conftest.py`).
- Add unit tests for new logic in `src/`; keep integration tests fast and deterministic.

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`). Example history: `feat: eliminate generic filler content`, `docs: update README`.
- PRs: Include clear description, linked issues, test plan (`pytest` output), and relevant screenshots/paths (e.g., `output/Your Deck_YYYYMMDD_HHMMSS.pptx`). Keep changes focused.

## Security & Configuration
- Never commit real API keys. Copy `.env.example` to `.env` and set `BRAVE_API_KEY`, `OPENAI_API_KEY` (+ optional LangSmith vars).
- Respect rate limits; do not add hard-coded secrets or print keys. Use `settings` and `dependencies` helpers.

## Architecture Notes
- LangGraph pipeline: research → load_docs → create_index → generate_outline → generate_content → create_presentation.
- Prefer adding new capabilities as tools in `src/tools.py` and wiring nodes in `src/deck_builder_agent.py`.
