"""
Shared cognee configuration for the Patient Health Memory demo.

IMPORTANT: import this module BEFORE importing cognee anywhere. Cognee reads
several settings from environment variables at import time, so this module
only touches os.environ and defines constants — it never imports cognee
itself.

"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

# ENABLE_BACKEND_ACCESS_CONTROL=true is load-bearing for patient isolation:
# with it OFF all datasets share one graph and recall(datasets=[pid])
# silently searches across patients (verified: patient B's scope answered
# from patient A's records). With it ON cognee keeps per-dataset databases
# and cross-patient questions correctly come back empty.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "true")
# Session-memory cache off: per-patient partitioning uses datasets, and a
# deterministic graph (documents only) demos better.
os.environ.setdefault("CACHING", "false")

# Keep all cognee storage inside the project instead of site-packages.
# Pre-create the directories: cognee's sqlite backend fails with "unable to
# open database file" if the parents don't exist on a fresh checkout.
# sqlite lives in <system_root>/databases and cognee doesn't create it itself
_COGNEE_DIR = PROJECT_ROOT / ".cognee"
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(_COGNEE_DIR / "data"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(_COGNEE_DIR / "system"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(_COGNEE_DIR / "cache"))
for _var in ("DATA_ROOT_DIRECTORY", "SYSTEM_ROOT_DIRECTORY", "CACHE_ROOT_DIRECTORY"):
    Path(os.environ[_var]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["SYSTEM_ROOT_DIRECTORY"]) / "databases").mkdir(exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not os.environ.get("LLM_API_KEY") and os.environ.get("OPENAI_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]

# text-embedding-3-small instead of cognee's default 3-large: low-tier
# OpenAI projects cap 3-large at 10k tokens/min, which throttles bulk
# ingestion into a crawl of 429-retries; 3-small has far higher limits and
# is plenty for this corpus. Dimensions auto-derive from the model name.
os.environ.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Per-patient partitioning (cognee 1.2.2 lifecycle API):
# The hackathon plan was remember(..., session_id=patient_id), but in the
# installed version session_id selects the *conversation session cache*
# (requires CACHING=true, bridges to the graph in the background) — it is
# NOT a content partition, and forget()/improve() don't accept it at all.
# The partition boundary that all four verbs share is the dataset:
#   remember(dataset_name=pid) / recall(datasets=[pid])
#   improve(dataset=pid)       / forget(dataset=pid)
# So patient_id == cognee dataset name here. Also noted in the README.
DEMO_PATIENT_ID = "anjali_deshpande"
DEMO_PATIENT_NAME = "Anjali Deshpande (synthetic demo patient)"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "patient_records"

# Kept for backwards compatibility with earlier scripts.
DATASET_NAME = DEMO_PATIENT_ID


def graph_html_path(patient_id: str) -> Path:
    return _COGNEE_DIR / f"graph_{patient_id}.html"


GRAPH_HTML_PATH = graph_html_path(DEMO_PATIENT_ID)


def require_llm_key() -> None:
    """Fail fast with a clear message if no LLM key is configured.

    Cognee 1.2.2 needs LLM_API_KEY for add(), cognify() and search()
    (default provider: OpenAI, also used for embeddings).
    """
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit(
            "LLM_API_KEY is not set. Export it or put it in .env "
            "(see .env.example). An OpenAI key covers both the LLM and "
            "embeddings with cognee's defaults."
        )


def document_paths() -> list[str]:
    paths = sorted(str(p) for p in DOCUMENTS_DIR.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No patient documents found in {DOCUMENTS_DIR}")
    return paths
