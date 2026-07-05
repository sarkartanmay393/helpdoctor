#!/usr/bin/env bash
# One-command runner for the Patient Health Memory demo.
#   ./run.sh            install deps, ingest (if needed), start the server
#   ./run.sh --reset    wipe cognee state first, then ingest fresh
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

echo "==> Installing dependencies (uv sync)"
uv sync

# The only required secret. Cognee's default provider is OpenAI; one OpenAI
# key covers both the LLM and embeddings. Never hardcode it — export it or
# put it in .env (see .env.example).
if [ -z "${LLM_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] \
   && ! grep -qsE "^(LLM_API_KEY|OPENAI_API_KEY)=." .env 2>/dev/null; then
  echo "ERROR: LLM_API_KEY is not set (OPENAI_API_KEY also accepted)."
  echo "  export LLM_API_KEY=sk-...   # or: cp .env.example .env and edit"
  exit 1
fi

if [ "${1:-}" = "--reset" ]; then
  echo "==> Resetting cognee state"
  uv run python reset.py
fi

echo "==> Ingesting patient records (skips if already ingested)"
uv run python ingest.py

echo "==> Starting server — open http://localhost:8000 in your browser"
uv run uvicorn server:app --host 127.0.0.1 --port 8000
