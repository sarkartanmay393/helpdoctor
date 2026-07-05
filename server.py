import asyncio
import os

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import DATASET_NAME, GRAPH_HTML_PATH, PROJECT_ROOT

import cognee
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ingest import dataset_exists, export_graph_html, run_ingestion
from queries import DEMO_QUESTIONS, ask_graph, ask_vector_baseline

app = FastAPI(title="Patient Health Memory")

# CORS wide open: local single-user demo, also covers opening index.html
# straight from the filesystem instead of via this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ingest_lock = asyncio.Lock()


class AskRequest(BaseModel):
    question: str
    compare: bool = True


class IngestRequest(BaseModel):
    force: bool = False


def _llm_key_missing_response() -> JSONResponse | None:
    if not os.environ.get("LLM_API_KEY"):
        return JSONResponse(
            status_code=503,
            content={"error": "LLM_API_KEY is not set on the server. "
                              "Export it (or add to .env) and restart."},
        )
    return None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/api/status")
async def status() -> dict:
    try:
        ingested = await dataset_exists()
    except Exception:
        ingested = False
    return {
        "llm_key_set": bool(os.environ.get("LLM_API_KEY")),
        "ingested": ingested,
        "graph_html_available": GRAPH_HTML_PATH.exists(),
        "demo_questions": DEMO_QUESTIONS,
    }


@app.post("/ingest")
async def ingest(body: IngestRequest | None = None):
    if err := _llm_key_missing_response():
        return err
    if _ingest_lock.locked():
        return JSONResponse(status_code=409,
                            content={"error": "Ingestion already running."})
    async with _ingest_lock:
        try:
            result = await run_ingestion(force=bool(body and body.force))
            return {"ok": True, **result}
        except SystemExit as exc:  # require_llm_key uses SystemExit
            return JSONResponse(status_code=503, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/ask")
async def ask(body: AskRequest):
    if err := _llm_key_missing_response():
        return err
    question = body.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Empty question."})
    if not await dataset_exists():
        return JSONResponse(
            status_code=409,
            content={"error": "No data ingested yet — run ingestion first."})
    try:
        if body.compare:
            # Run the graph answer and the vector-only baseline concurrently;
            # the baseline failing must never sink the real answer.
            graph_res, baseline_res = await asyncio.gather(
                ask_graph(question),
                ask_vector_baseline(question),
                return_exceptions=True,
            )
            if isinstance(graph_res, BaseException):
                raise graph_res
            answer, sources = graph_res
            baseline = (None if isinstance(baseline_res, BaseException)
                        else baseline_res)
        else:
            answer, sources = await ask_graph(question)
            baseline = None
        return {
            "question": question,
            "graph_answer": answer,
            "vector_answer": baseline,
            "sources": sources,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/graph")
async def graph(refresh: bool = False):
    if (refresh or not GRAPH_HTML_PATH.exists()) and await dataset_exists():
        await export_graph_html()
    if not GRAPH_HTML_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Graph not exported yet — run ingestion first."})
    return FileResponse(GRAPH_HTML_PATH, media_type="text/html")
