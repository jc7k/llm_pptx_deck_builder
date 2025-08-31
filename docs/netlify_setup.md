# Netlify Deployment Setup

This guide walks through configuring Netlify and GitHub so the new web UI can trigger deck builds and serve the resulting PPTX files.

## Prerequisites
- Netlify account (Starter/free is fine)
- GitHub repository access (this repo)

## 1) GitHub Secrets for Workflow
No provider API keys are stored in the repository. Users supply their own `OPENAI_API_KEY` and `BRAVE_API_KEY` per request. You do not need to add these as repo secrets.
If you use optional tracing/observability providers, add only those non-sensitive keys as needed.

## 2) GitHub Personal Access Token (PAT) for Netlify
Create a fine-scoped PAT (classic is acceptable):
- Scopes: `repo`, `workflow`
- Keep this token only in Netlify env vars (not in GitHub)

## 3) Netlify Site
1. Create a new site from Git (select this repo).
2. Build settings will be picked up from `netlify.toml`.

## 4) Netlify Environment Variables
Add these under Site settings → Environment variables:
- `GITHUB_TOKEN`: the PAT with `repo` + `workflow` scopes
- `GITHUB_OWNER`: GitHub org/user (e.g. `your-org`)
- `GITHUB_REPO`: repository name (e.g. `llm_pptx_deck_builder`)
- `GITHUB_WORKFLOW`: `build_deck.yml` (default matches repo file)
- `GITHUB_BRANCH`: `main` (or your default branch)

## 5) Deploy
- Push the feature branch and open a PR. Netlify will create a Deploy Preview.
- Use the preview URL to test the flow: enter a topic → submit → watch status → download PPTX.

## 6) Troubleshooting
- 401/403 from functions: confirm `GITHUB_TOKEN` scope, owner/repo names, and that the workflow file exists on the target branch.
- `not_found` status: it can take a few seconds before the run appears; polling continues. Verify `run-name` format is `deck-<request_id>`.
- No PPTX in artifact: check Actions logs for CLI errors and ensure the deck was saved under `output/`. Ensure tokens are valid and not rate-limited.
- Rate limits: reduce polling rate (6–8s) or try again later.

## 7) Local Development
- Frontend: `cd web && npm install && npm run dev` (functions are not available locally without Netlify CLI).
- Functions: optional `netlify dev` can proxy functions locally if you install Netlify CLI.

## 8) Cleanup & Governance
- Rotate the GitHub PAT periodically; store it only in Netlify.
- Consider setting GitHub branch protection on `build_deck.yml` if needed.
- Ensure artifact retention matches your needs (default GitHub retention applies).
