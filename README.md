# Patient Health Memory

**The pitch (4 sentences):** In Indian healthcare, a patient's records are
scattered across hospitals, family physicians, labs, and specialists in
different cities, with no continuity of care — no single doctor ever sees the
whole story. Patient Health Memory ingests those scattered documents into a
[Cognee](https://github.com/topoteretes/cognee) knowledge graph and lets the
patient ask natural-language questions across their entire history. Cognee's
graph layer is the load-bearing part, not just "we used an LLM": entities like
a doctor, an admission, or a diagnosis become shared nodes across documents,
so the system answers questions that pure vector RAG provably cannot — e.g.
naming the condition a prescription treats when the prescription only points
to a discharge summary from another hospital. Everything is open-source and
self-hosted: local LanceDB vectors, an embedded graph database, no Docker, no
external services — targeting the open-source / self-hosted track.

All patient data is **synthetic and clearly fictional** (headers in every file).

## Quickstart

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set the one required secret (an OpenAI key covers LLM + embeddings)
export LLM_API_KEY=sk-...        # or: cp .env.example .env and edit

# 3. Install deps, ingest the 5 records, start the server
./run.sh

# 4. Open the demo
open http://localhost:8000
```

Ingestion makes LLM calls and takes a few minutes on first run; re-runs are
skipped automatically. `./run.sh --reset` wipes and rebuilds from scratch.

## Manual commands

```bash
uv sync                              # install dependencies
uv run python ingest.py             # add() + cognify(temporal_cognify=True)
uv run python ingest.py --force     # wipe + rebuild
uv run python queries.py --compare  # 5 demo questions, graph vs vector, in terminal
uv run python reset.py              # prune_data() + prune_system()
uv run uvicorn server:app --port 8000
```

## The demo, in 30 seconds

Five fictional documents follow one patient across 18 months and three cities:
a Pune hospital discharge summary (names the diagnosis — the only document
that does), a Mumbai lab report, a referral letter to a Chennai specialist,
the specialist's prescription (names the drug, **not** the condition), and a
follow-up note. Ask *"What condition is the apixaban treating?"* — vector
search retrieves only the prescription chunk, which deliberately never names
the condition; the knowledge graph walks
`apixaban → rhythm disorder → January 2025 discharge summary → atrial
fibrillation` and answers correctly. The UI shows the graph answer and the
vector-only RAG baseline side by side, plus the live knowledge graph the
answers traverse.

## Architecture

- **Cognee 1.2.2** — `add()` → `cognify(temporal_cognify=True)` → `search()`.
  Graph answers use `SearchType.GRAPH_COMPLETION`; the baseline column uses
  `SearchType.RAG_COMPLETION` (plain vector retrieval, no graph).
- **Storage, all in-process:** LanceDB (vectors), Ladybug (embedded graph DB —
  cognee 1.2.2 no longer ships a NetworkX provider, Ladybug is the embedded
  default), SQLite (metadata). Everything lives in `./.cognee/`.
- **Backend:** FastAPI ([server.py](server.py)) — `POST /ingest`, `POST /ask`,
  `GET /graph` (cognee's exported visualization, embedded in the UI).
- **Frontend:** one static [index.html](static/index.html), vanilla JS, no build step.
