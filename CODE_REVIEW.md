# Code Review Report

This document summarizes issues identified during a blind review of the repository, along with rationale and patches. Changes applied in this pass are conservative and avoid altering core behavior; additional suggestions are provided for future improvement.

## Summary of Applied Fixes

- Align tests and code: `search_web` now uses `requests.get` with retry/backoff for easier mocking in tests.
- Security default: Web loader now verifies SSL by default and only disables verification if configured.
- Robustness: Safer handling of PowerPoint placeholders (subtitle/agenda/content) to avoid index errors with custom templates.
- Stability: Defensive `chunk_count` calculation for LlamaIndex (private attributes may change upstream).
- Test compatibility: `create_vector_index` now returns metadata and includes the direct `_index` handle expected by tests.
- Cleanup: Simplified quote trimming in slide title optimization.
- Minor: Deduplicate extracted URLs.

## Detailed Findings and Rationale

### 1) Test/code mismatch in web search
- File: `src/tools.py` (`search_web`)
- Issue: Tests patch `requests.get`, but the code used `Session.get()`, so mocks wouldn’t intercept calls.
- Impact: Flaky or failing tests; unnecessary complexity for contributors.
- Patch: Use `requests.get` with existing manual retry/backoff. Behavior is unchanged for users.

### 2) Insecure default in web document loading
- File: `src/tools.py` (`load_web_documents`)
- Issue: `verify=False` disabled SSL verification globally; also suppressed warnings unconditionally.
- Risk: Security and integrity; unexpected acceptance of MITM traffic.
- Patch: Add `verify_ssl` setting (default True). Only disable warnings when verification is off.

### 3) Fragile reliance on private LlamaIndex internals
- File: `src/tools.py` (`create_vector_index`)
- Issue: Access to `index.vector_store._data.embedding_dict` may break with upstream changes.
- Patch: Wrap in `try/except` and fall back to `0` if unavailable.

### 4) Tests expect `_index` in vector index metadata
- File: `src/tools.py` (`create_vector_index`)
- Issue: Tests look for `"_index" in result`; code returned only metadata.
- Patch: Return `{**metadata, "_index": index}`. Core behavior (storing by `index_id`) remains.

### 5) Overly complex string stripping for optimized titles
- File: `src/tools.py` (`optimize_slide_title`)
- Issue: Non-obvious quote stripping using concatenated literals.
- Patch: Replace with `strip(" \t\n\r\"'")` for clarity.

### 6) Placeholder access can raise on custom templates
- File: `src/tools.py` (`create_presentation`)
- Issue: Direct indexing into `placeholders[1]` may fail depending on layout.
- Patch: Wrap in `try/except` for subtitle, agenda, and content placeholders.

### 7) Duplicate URLs from search results
- File: `src/tools.py` (`extract_urls_from_search_results`)
- Issue: No deduping; redundant loads waste time and quota.
- Patch: Track `seen` set to deduplicate while preserving order.

## Additional Opportunities (Not Applied)

- Logging vs print: Replace `print` with `logging` across tools for configurable verbosity.
- Session reuse: Maintain a module-level `requests.Session` for connection pooling and retries; tests can mock via a small wrapper (e.g., `http_get`).
- Memory hygiene: `_vector_index_store` can grow unbounded. Consider LRU eviction (e.g., keep last N indexes).
- CLI flags: `--max-slides`, `--audience`, `--duration` are currently unused; wire them into outline/content prompts or remove to reduce confusion.
- Template robustness: Consider detecting placeholders by type rather than numeric index for broader template compatibility.
- Security headers: Consider rate limiting and user-agent policies in README/CONTRIBUTING for scraping etiquette.

## Patches Applied (References)

- `src/tools.py`
  - `search_web`: switch to `requests.get` with backoff.
  - `load_web_documents`: honor `settings.verify_ssl`; conditional warning suppression.
  - `create_vector_index`: defensive `chunk_count`; include `_index` in return.
  - `optimize_slide_title`: simplify quote stripping.
  - `create_presentation`: guard placeholder access.
  - `extract_urls_from_search_results`: deduplicate URLs.
- `src/settings.py`
  - Add `verify_ssl: bool = True` with description.

All changes are minimal, backward-compatible, and focused on correctness and maintainability.
