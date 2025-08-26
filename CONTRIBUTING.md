# Contributing Guide

Thanks for your interest in improving llm_pptx_deck_builder! This guide explains how to set up your environment, make changes, and propose them for review.

## Getting Started
- Requirements: Python 3.11+, uv (package manager), git.
- Setup:
  - `git clone <repo> && cd llm_pptx_deck_builder`
  - `uv venv && uv sync`
  - `cp .env.example .env` and populate `BRAVE_API_KEY`, `OPENAI_API_KEY` (do not commit secrets).

## Development Commands
- Run CLI: `uv run python deck_builder_cli.py --topic "AI in 2025" --verbose`
- Validate config: `uv run python deck_builder_cli.py --validate-only`
- Tests: `uv run pytest -v` (e.g., `-k tools`, `-m integration`)
- Lint: `uv run ruff check src tests`
- Format: `uv run black .`
- Type-check: `uv run mypy src`

If you add dependencies, update `pyproject.toml`, run `uv sync`, and commit both `pyproject.toml` and `uv.lock`.

## Code Style
- Python: Black formatting, Ruff linting, MyPy types for new/changed code.
- Use 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep docstrings concise; prefer small, focused functions.
- Never print or log API keys; read config via `src/settings.py`/`src/dependencies.py`.

## Tests
- Place tests in `tests/` as `test_*.py`; follow patterns in `pytest.ini`.
- Mock network/LLM/IO (see fixtures in `tests/conftest.py` and `unittest.mock.patch`).
- Add unit tests for new logic and update integration tests if behavior changes.
- PRs should pass `pytest`, `ruff`, `black --check`, and `mypy` locally.

## Architecture Tips
- Workflow nodes live in `src/deck_builder_agent.py` (LangGraph).
- Tools (search, loading, indexing, content, pptx) live in `src/tools.py`.
- CLI flags in `deck_builder_cli.py`; config in `src/settings.py` and `src/dependencies.py`.
- Prefer adding new capabilities as tools, then wire nodes/edges in the graph.

## Commits & PRs
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` (history shows similar style).
- Branch names: `feat/<short-topic>`, `fix/<short-topic>`.
- Open small, focused PRs with:
  - Clear description and rationale; link issues.
  - Test plan (commands + expected output).
  - Screenshots or output path if relevant (e.g., `output/*.pptx`).
  - No generated binaries or secrets committed.

## Reporting Issues & Security
- Include steps to reproduce, expected vs. actual, logs (sanitized), and environment info.
- For potential security issues, avoid public details; share only necessary information and do not include secrets.

By contributing, you agree your changes are licensed under the repository’s MIT license.
