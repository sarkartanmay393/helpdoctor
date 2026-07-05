import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import (
    COGNEE_CLOUD_ENABLED,
    DEMO_PATIENT_ID,
    PROJECT_ROOT,
    graph_html_path,
)

import cognee
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import registry
from ingest import (
    dataset_exists,
    ensure_db_setup,
    export_graph_html,
    remember_text,
    run_ingestion,
)
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

ALLOWED_DOC_EXTENSIONS = {".txt", ".pdf"}


class AskRequest(BaseModel):
    question: str
    patient_id: str = DEMO_PATIENT_ID
    compare: bool = True

class ForgetRequest(BaseModel):
    patient_id: str
    
class TranscriptRequest(BaseModel):
    patient_id: str
    patient_name: str = ""
    text: str

class DoctorAskRequest(BaseModel):
    question: str


def _llm_key_missing_response() -> JSONResponse | None:
    if COGNEE_CLOUD_ENABLED:
        return None  # cloud mode: the hosted side brings its own LLM
    if not os.environ.get("LLM_API_KEY"):
        return JSONResponse(
            status_code=503,
            content={"error": "LLM_API_KEY is not set on the server. "
                              "Export it (or add to .env) and restart."},
        )
    return None


def _clean_patient_id(raw: str) -> str:
    pid = "".join(ch if ch.isalnum() else "_" for ch in raw.strip().lower())
    return pid.strip("_")


_ocr_engine = None
_OCR_MAX_PAGES = 20  # cap pathological documents; a record is rarely longer


def _ocr_pdf(data: bytes) -> str:
    """OCR fallback for scanned/image-only PDFs: rasterize pages with
    pypdfium2 (already here as a pdfplumber dependency) and read them with
    RapidOCR — local ONNX models, offline, no system binaries."""
    global _ocr_engine
    import io

    import numpy as np
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    pdf = pdfium.PdfDocument(io.BytesIO(data))
    pages_text: list[str] = []
    try:
        for i, page in enumerate(pdf):
            if i >= _OCR_MAX_PAGES:
                break
            bitmap = page.render(scale=200 / 72)  # ~200 DPI, plenty for print
            image = np.asarray(bitmap.to_pil().convert("RGB"))
            result, _ = _ocr_engine(image)
            if result:
                pages_text.append("\n".join(item[1] for item in result))
    finally:
        pdf.close()
    return "\n\n".join(pages_text)


def _pdf_text(data: bytes) -> str:
    """Three extraction stages, because real-world PDFs are messy: pypdf
    first (fast), pdfplumber second (pdfminer-based, reads many PDFs whose
    text layer pypdf returns empty for), and local OCR last for scanned
    image-only PDFs — the common case for Indian medical records."""
    import io

    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # PDFs "encrypted" with an empty owner password
            except Exception:
                raise ValueError(
                    "This PDF is password-protected. Remove the password and "
                    "re-upload.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except ValueError:
        raise
    except Exception:
        pass  # fall through to pdfplumber

    if not text.strip():
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            pass

    if not text.strip():
        try:
            text = _ocr_pdf(data)
        except Exception:
            pass

    if not text.strip():
        raise ValueError(
            "Could not read this PDF: no text layer, and OCR found no "
            "readable text either (is the scan legible?). Please upload a "
            "clearer scan or a text-based PDF, or copy the content into a "
            ".txt file and upload that instead.")
    return text


def _extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _pdf_text(data)
    raise ValueError(f"Unsupported file type: {ext} (allowed: .txt, .pdf)")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/api/status")
async def status() -> dict:
    try:
        demo_ingested = await dataset_exists(DEMO_PATIENT_ID)
    except Exception:
        demo_ingested = False
    return {
        "llm_key_set": bool(os.environ.get("LLM_API_KEY")) or COGNEE_CLOUD_ENABLED,
        "mode": "cloud" if COGNEE_CLOUD_ENABLED else "local",
        "demo_patient_id": DEMO_PATIENT_ID,
        "demo_ingested": demo_ingested,
        "demo_questions": DEMO_QUESTIONS,
    }


@app.get("/patients")
async def patients() -> list[dict]:
    return registry.list_patients()


@app.get("/patients/{patient_id}/documents")
async def patient_documents(patient_id: str) -> list[dict]:
    return registry.list_documents(patient_id)


@app.post("/ingest")
async def ingest_document(
    patient_id: str = Form(...),
    patient_name: str = Form(default=""),
    file: UploadFile = File(...),
):
    if err := _llm_key_missing_response():
        return err
    pid = _clean_patient_id(patient_id)
    if not pid:
        return JSONResponse(status_code=400, content={"error": "Empty patient id."})
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{ext}'. Upload .txt or .pdf."})

    data = await file.read()
    digest = registry.content_hash(data)
    registry.ensure_patient(pid, patient_name.strip() or None)

    # Hash-based dedup: never remember() the same bytes twice for a patient.
    if registry.document_exists(pid, digest):
        return {"ok": True, "duplicate": True,
                "message": "This document was already added for this patient — "
                           "skipped (nothing re-ingested)."}

    try:
        # In a thread: PDF parsing is blocking, and the OCR fallback for
        # scanned PDFs takes a few seconds per page.
        text = await asyncio.to_thread(_extract_text, file.filename or "upload", data)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if not text.strip():
        return JSONResponse(status_code=400,
                            content={"error": "No text could be extracted."})

    if _ingest_lock.locked():
        return JSONResponse(status_code=409,
                            content={"error": "Another ingestion is running — retry shortly."})
    async with _ingest_lock:
        try:
            result = await remember_text(pid, text, file.filename or "upload",
                                         digest=digest)
            return {"ok": True, "duplicate": False, "patient_id": pid, **result}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/ingest_transcript")
async def ingest_transcript(body: TranscriptRequest):
    """Pasted conversation transcript — same pipeline as the document upload, just a different text source."""
    
    if err := _llm_key_missing_response():
        return err
    pid = _clean_patient_id(body.patient_id)
    if not pid:
        return JSONResponse(status_code=400, content={"error": "Empty patient id."})
    text = body.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Empty transcript."})

    # Hash the pasted text exactly like uploaded file bytes, so an accidental
    # duplicate paste hits the same registry dedup as a duplicate upload.
    digest = registry.content_hash(text.encode())
    registry.ensure_patient(pid, body.patient_name.strip() or None)
    if registry.document_exists(pid, digest):
        return {"ok": True, "duplicate": True,
                "message": "This transcript was already added for this patient — "
                           "skipped (nothing re-ingested)."}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"conversation-transcript-{stamp}.txt"
    labeled = f"Transcript of a doctor-patient conversation ({filename}):\n{text}"

    if _ingest_lock.locked():
        return JSONResponse(status_code=409,
                            content={"error": "Another ingestion is running — retry shortly."})
    async with _ingest_lock:
        try:
            result = await remember_text(pid, labeled, filename, digest=digest)
            return {"ok": True, "duplicate": False, "patient_id": pid, **result}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/ingest_demo")
async def ingest_demo():
    if err := _llm_key_missing_response():
        return err
    if _ingest_lock.locked():
        return JSONResponse(status_code=409,
                            content={"error": "Ingestion already running."})
    async with _ingest_lock:
        try:
            result = await run_ingestion(force=False)
            return {"ok": True, **result}
        except SystemExit as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/ask")
async def ask(body: AskRequest):
    if err := _llm_key_missing_response():
        return err
    question = body.question.strip()
    pid = _clean_patient_id(body.patient_id) or DEMO_PATIENT_ID
    if not question:
        return JSONResponse(status_code=400, content={"error": "Empty question."})
    if not await dataset_exists(pid):
        return JSONResponse(
            status_code=409,
            content={"error": f"No data ingested yet for patient '{pid}'."})
    try:
        if body.compare:
            # Graph answer and vector-only baseline run concurrently; the
            # baseline failing must never sink the real answer.
            graph_res, baseline_res = await asyncio.gather(
                ask_graph(question, pid),
                ask_vector_baseline(question, pid),
                return_exceptions=True,
            )
            if isinstance(graph_res, BaseException):
                raise graph_res
            answer, sources = graph_res
            baseline = (None if isinstance(baseline_res, BaseException)
                        else baseline_res)
        else:
            answer, sources = await ask_graph(question, pid)
            baseline = None
        return {
            "question": question,
            "patient_id": pid,
            "graph_answer": answer,
            "vector_answer": baseline,
            "sources": sources,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/doctor/ask")
async def doctor_ask(body: DoctorAskRequest):
    """Global doctor-facing chat. Deterministic routing (see doctor_desk.py);
    only the recall route touches the LLM. The route name is returned so the
    UI can show which path answered."""
    from doctor_desk import route_question

    question = body.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Empty question."})
    routed = route_question(question)
    if routed["route"] != "recall":
        return routed

    if err := _llm_key_missing_response():
        return err
    pid = routed["patient_id"]
    if not await dataset_exists(pid):
        return {"route": "recall", "patient_id": pid,
                "answer": f"No data has been ingested into memory for '{pid}' yet."}
    try:
        answer, sources = await ask_graph(question, pid)
        return {"route": "recall", "patient_id": pid,
                "answer": answer, "sources": sources}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/forget")
async def forget(body: ForgetRequest):
    """Lifecycle verb 4: patient-initiated deletion of their memory graph."""
    
    pid = _clean_patient_id(body.patient_id)
    if not pid:
        return JSONResponse(status_code=400, content={"error": "Empty patient id."})
    try:
        try:
            if COGNEE_CLOUD_ENABLED:
                from cloud import cloud_forget

                result = await cloud_forget(pid)
            else:
                await ensure_db_setup()
                result = await cognee.forget(dataset=pid)
        except Exception as exc:
            # Dataset may not exist in cognee (e.g. registry-only patient).
            result = f"(cognee dataset not deleted: {exc})"
        removed = registry.forget_patient(pid)
        html = graph_html_path(pid)
        if html.exists():
            html.unlink()
        return {"ok": True, "patient_id": pid,
                "forget_result": str(result),
                "registry_rows_removed": removed}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/graph")
async def graph(patient_id: str = DEMO_PATIENT_ID, refresh: bool = False):
    if COGNEE_CLOUD_ENABLED:
        # The HTML export renders from the local graph database; in cloud
        # mode the graph lives on the hosted instance.
        return JSONResponse(
            status_code=404,
            content={"error": "Graph visualization is available in local "
                              "mode only (COGNEE_CLOUD=false)."})
    pid = _clean_patient_id(patient_id) or DEMO_PATIENT_ID
    html = graph_html_path(pid)
    if (refresh or not html.exists()) and await dataset_exists(pid):
        await export_graph_html(pid)
    if not html.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"No graph exported for '{pid}' — ingest data first."})
    return FileResponse(html, media_type="text/html")
