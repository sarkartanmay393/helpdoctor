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

# Single-user local demo: disable cognee 1.x multi-tenant access control and
# the session-memory layer (both on by default) so the graph contains only
# the patient documents and every run is deterministic.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
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

DATASET_NAME = "patient_records"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "patient_records"
GRAPH_HTML_PATH = _COGNEE_DIR / "graph.html"


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
