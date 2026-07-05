# Patient Health Memory

**The problem.** In Indian healthcare, a patient's records are scattered
across hospitals, family physicians, labs, and specialists in different
cities, with no continuity of care — a doctor's office starts every visit
with an incomplete picture, and the patient has no usable access to their
own history.

**The solution.** Patient Health Memory is a **memory layer for a doctor's
office**, built on [Cognee](https://github.com/topoteretes/cognee). The
office records everything about a patient in one place: after each visit,
the front desk uploads incoming documents (discharge summaries, lab
reports, prescription PDFs — scanned ones are OCR'd locally, since Indian
medical records are usually scans) and the consultation transcript, and
each one is ingested into that patient's own knowledge graph. In production, the
transcripts come from speech-to-text on doctor-patient audio; this demo
shows the ingestion path directly via a pasted transcript. Inside the
office, a **Doctor's Desk** chat answers questions across the whole patient
registry. And the office exposes a **patient-access API**: each patient can
query their own records in plain language through endpoints strictly scoped
to their own graph, and can have the office erase their data with one
`forget()` call.

**Why Cognee's graph layer specifically.** Entities like a doctor, an
admission, or a diagnosis become shared graph nodes across documents from
different providers, so the system answers multi-hop questions that
single-chunk vector retrieval provably cannot — e.g. naming the condition a
prescription treats when the prescription only points to a discharge summary
from another hospital, in another city. The UI shows the graph answer and a
vector-only RAG baseline side by side so the difference is visible, not
claimed.

**Track: Open Source.** Everything is open-source and self-hosted: local
LanceDB vectors, an embedded graph database, SQLite — no Docker, no external
services, one API key.

All patient data is **synthetic and clearly fictional** (headers in every file).

## Quickstart

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Set the one required secret (an OpenAI key covers LLM + embeddings)
export LLM_API_KEY=sk-...        # or: cp .env.example .env and edit

# 3. Install deps, seed the demo patient, start the server
./run.sh

# 4. Open the demo
open http://localhost:8000
```

Seeding makes LLM calls and takes a few minutes on first run; re-runs are
deduplicated automatically. `./run.sh --reset` wipes and rebuilds.

## Manual commands

```bash
uv sync                              # install dependencies
uv run python ingest.py             # remember() the 9 demo records + improve()
uv run python ingest.py --force     # wipe + rebuild
uv run python queries.py --compare  # 5 demo questions, graph vs vector baseline
uv run python reset.py              # full wipe (prune + registry)
uv run python reset.py --patient anjali_deshpande   # forget() one patient
uv run uvicorn server:app --port 8000
```

## The Cognee lifecycle API (and one honest note)

The pipeline uses all four lifecycle verbs of cognee 1.2.2:

- **`remember(files_or_text, dataset_name=patient_id, temporal_cognify=True)`**
  — per-patient ingestion (add + cognify under the hood, with temporal
  extraction so before/after questions work).
- **`recall(question, datasets=[patient_id], query_type=GRAPH_COMPLETION)`**
  — patient-scoped question answering; the vector baseline is the same call
  with `RAG_COMPLETION` and `top_k=3`.
- **`improve(dataset=patient_id)`** — called explicitly after every ingestion
  (with `self_improvement=False` on remember, so the refinement pass is a
  single visible, logged step).
- **`forget(dataset=patient_id)`** — wired to the "Forget this patient's
  data" button in the UI: when a patient asks the office to delete their
  data, one call erases their entire graph.

**Note on `session_id`:** the original plan was to partition patients by
`remember(..., session_id=...)`, but in the installed cognee 1.2.2,
`session_id` selects the *conversation session cache* (a fast recall layer
that bridges to the graph in the background) — it is not a content
partition, and `forget()`/`improve()` don't accept it at all. The partition
boundary all four verbs share is the **dataset**, so here
`patient_id == dataset name`. Verified against the installed package source,
not the docs.

**Note on isolation:** dataset scoping is only real with
`ENABLE_BACKEND_ACCESS_CONTROL=true` (set in [cognee_setup.py](cognee_setup.py)).
With it off, all datasets share one graph and we verified patient B's
questions were answered from patient A's records; with it on, each dataset
gets its own databases and cross-patient questions correctly return nothing.

## The demo, in 30 seconds

Nine fictional documents follow one patient across 18 months and three
cities — five that tell a connected cardiac story and four realistic
distractors (dental, eye, orthopaedic, wellness). The Pune discharge summary
is the only document that names the diagnosis; the Chennai prescription names
the drug but deliberately not the condition. Ask *"What condition is the
apixaban treating?"* — top-k vector retrieval fetches the prescription and
its neighbours, none of which name the condition; the knowledge graph walks
`apixaban → rhythm disorder → January 2025 discharge summary → atrial
fibrillation` and answers correctly, with sources.

## Architecture

- **Cognee 1.2.2** lifecycle API (see above), one dataset per patient.
- **Storage, all in-process:** LanceDB (vectors), Ladybug (embedded graph DB —
  cognee 1.2.2 no longer ships a NetworkX provider, Ladybug is the embedded
  default), SQLite (cognee metadata + the patient registry). Everything
  lives in `./.cognee/` and `./patients.db`.
- **Patient registry** ([registry.py](registry.py)): a lookup index only
  (who exists, which file hashes were ingested — dedup). Medical content
  lives in Cognee; the registry is never a fallback source of truth.
- **Doctor's Desk** ([doctor_desk.py](doctor_desk.py)): a doctor-facing chat
  across the registry with deterministic routing — registry SQL lookups and a
  hardcoded demo schedule answer instantly without an LLM; free-form clinical
  questions about a named patient route to `recall()` scoped to that patient.
  No booking, no calendar, no cross-patient aggregate reasoning.
- **Backend:** FastAPI ([server.py](server.py)). Office-side endpoints:
  `/ingest` (multipart, hash-dedup; PDF text via pypdf → pdfplumber →
  RapidOCR fallback for scanned image-only PDFs, all local/offline),
  `/ingest_transcript` (pasted conversation, same pipeline + dedup),
  `/doctor/ask`, `/patients`.
  Patient-access API (each call scoped to one patient's graph):
  `/ask`, `/patients/{id}/documents`, `/graph?patient_id=...`, `/forget`.
  The isolation behind that scoping is enforced by cognee's access control
  (see the note above) and was verified empirically.
- **Frontend:** one static [index.html](static/index.html), vanilla JS, no build step.
