# Netlify Web UI for Deck Builder

This document describes the architecture, project plan, and task breakdown to add a Netlify‑deployable web UI that triggers deck generation safely without exposing API keys.

## Goals
- Provide a simple web form to request a deck and download the resulting `.pptx`.
- Keep secrets off the client: use Netlify Functions to talk to GitHub.
- Run heavy work in GitHub Actions, reusing the existing Python CLI.
- Fit within Netlify Starter (free) limits.

---

## Architecture

### Overview
- Frontend: Static site (`web/`) built with Vite + React (TypeScript).
- API: Netlify Functions (`netlify/functions/`) that call GitHub REST API.
- Worker: GitHub Actions workflow that runs the Python CLI and uploads an artifact.

### Flow
1. User enters topic and their own API keys in the form. The UI generates a `request_id` (UUID). Keys are held in-memory only and never persisted.
2. UI calls `/.netlify/functions/create_job` with `{ request_id, topic, slide_count, openai_api_key, brave_api_key }`.
3. Netlify function triggers GitHub Actions via `repository_dispatch` (`event_type: deck-request`) including the payload. The workflow sets `run-name: deck-${request_id}`. Tokens are never logged and are masked.
4. UI polls `/.netlify/functions/job_status?request_id=...` to track the run.
5. On success, UI calls `/.netlify/functions/download_artifact?request_id=...` to fetch the PPTX.
6. Function downloads the GitHub artifact, extracts the `.pptx`, and streams it back to the browser.

### Components
- `web/` (static frontend)
  - `App.tsx`: form (topic, optional slide count), submission + status, download button.
  - `api.ts`: client for Netlify Functions.
- `netlify/functions/` (serverless API)
  - `create_job.ts`: dispatch workflow on GitHub with inputs.
  - `job_status.ts`: resolve workflow run by `run-name` and report status.
  - `download_artifact.ts`: fetch artifact ZIP, extract `.pptx`, return file.
  - `package.json`: function-only dependencies (e.g., `adm-zip`).
- `.github/workflows/build_deck.yml` (worker)
  - Trigger: `repository_dispatch` with `types: [deck-request]` and `client_payload` (`request_id`, `topic`, `slide_count`, `openai_api_key`, `brave_api_key`).
  - Steps: checkout → mask tokens → setup Python (3.11) → setup `uv` → `uv sync` → run CLI (env uses client payload tokens) → upload artifact named `deck-${request_id}`.
- `netlify.toml`
  - Build `web/`, publish `web/dist`, functions in `netlify/functions/`.

### Secrets & Configuration
- Netlify environment variables:
  - `GITHUB_TOKEN`: PAT with `repo` + `workflow` scopes.
  - `GITHUB_OWNER`, `GITHUB_REPO`: repository coordinates.
  - `GITHUB_WORKFLOW`: `build_deck.yml`.
  - `GITHUB_BRANCH`: default `main` (or your default branch).
- BYO user tokens are provided at request time and never stored server-side or in the repo. The workflow masks them and they are not included in run inputs.

### Free Tier Fit
- Static hosting + short Netlify Functions (small payloads, low CPU).
- All heavy compute and bandwidth (artifact generation) occurs in GitHub Actions and GitHub artifact storage.
- Polling (every 4–6s) keeps each function call fast and well under free quotas.

---

## Project Plan

1. Docs: Architecture, plan, and setup instructions.
2. Frontend scaffold: Vite + React + TS in `web/` with minimal form and status UI.
3. Functions: Implement `create_job`, `job_status`, `download_artifact` with GitHub API.
4. Actions workflow: Add `build_deck.yml` to run CLI and upload artifact.
5. Configuration: `netlify.toml`, set Node 18 for functions.
6. Wiring: Connect UI to functions; implement polling and download.
7. Docs & handoff: Setup guide (secrets, deploy, local dev), PR checklist.

---

## Detailed Task List

### 1) Documentation
- Add this architecture & plan to `docs/netlify_web_ui.md`.
- Add a short README section for Web UI setup (link to this doc).

### 2) Frontend (web/)
- Create Vite React TS app structure (no runtime install required here).
- Add `index.html`, `tsconfig.json`, `vite.config.ts`, `src/main.tsx`, `src/App.tsx`.
- Implement form: fields for `topic` (+ optional `slide_count`).
- Generate `request_id` with `crypto.randomUUID()` on submit.
- Call functions via `api.ts`; show status (queued/running/success/failure).
- Enable file download on success using `download_artifact` endpoint.

### 3) Netlify Functions
- Create `netlify/functions/package.json` with dependencies: `adm-zip`.
- `create_job.ts`
  - Validate input, construct `repository_dispatch` payload with client payload including keys.
  - Return 202 + `{ request_id }` and do not echo internal errors.
- `job_status.ts`
  - Lookup latest run for the workflow filtered by `event=repository_dispatch` and `run-name` `deck-${request_id}`.
  - Return `{ status, conclusion, run_id }`.
- `download_artifact.ts`
  - Using `run_id` or `request_id`, locate artifact `deck-${request_id}` from that run.
  - Download ZIP, extract `.pptx`, return as binary with proper headers.

### 4) GitHub Actions Workflow
- Add `.github/workflows/build_deck.yml` with `repository_dispatch` type `deck-request`.
- Set `run-name: deck-${{ github.event.client_payload.request_id }}`.
- Steps: checkout → mask tokens → setup Python 3.11 → setup `uv` → `uv sync` → run CLI using payload env → find latest `.pptx` → copy to `artifact/deck-${request_id}.pptx` → upload with `actions/upload-artifact@v4`.

### 5) Netlify Config
- Add `netlify.toml` with:
  - `[build] command = "npm --prefix web ci && npm --prefix web run build"`
  - `publish = "web/dist"`, `functions = "netlify/functions"`.
  - `node_bundler = "esbuild"` for functions (default).

### 6) Validation & UX
- Basic error states (failed run, missing artifact, rate limit messages).
- Minimal styling, accessible form.

### 7) Delivery
- Open PR `feat/netlify-ui` with Deploy Preview.
- Include setup instructions and environment variable checklist.

---

## Risks & Mitigations
- Token exposure in logs: mask tokens and avoid printing command env. Avoid using workflow inputs to carry tokens.
- Artifact ZIP extraction in function: use `adm-zip` to avoid manual parsing.
- GitHub PAT scope: use least privilege; rotate if leaked; store only in Netlify.
- Build minutes on GitHub: public repos are free; private repos consume minutes.
- API rate limits: keep polling to 4–6s; backoff on rate limit headers.

## Acceptance Criteria
- Submitting a topic triggers a workflow run.
- Status transitions: queued → in_progress → completed (success/failure).
- On success, the browser downloads a `.pptx` file.
- No client-side secrets; keys are only in Netlify/GitHub.
