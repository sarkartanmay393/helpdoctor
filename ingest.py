"""Ingestion pipeline built on cognee's lifecycle API (remember/improve).

Per-patient partitioning uses dataset_name=patient_id — see the comment in
cognee_setup.py for why session_id is not the content partition in the
installed cognee 1.2.2.

Flow per document: registry hash-dedup -> remember() -> registry record.
After new content lands: explicit improve() (logged) + graph HTML export.
"""

import asyncio
import sys
import time

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import (
    DEMO_PATIENT_ID,
    DEMO_PATIENT_NAME,
    DOCUMENTS_DIR,
    document_paths,
    graph_html_path,
    require_llm_key,
)

import cognee
import registry


async def ensure_db_setup() -> None:
    """Create cognee's relational schema on first run (idempotent).

    Without this, any cognee call that resolves the default user raises
    DatabaseNotCreatedError on a fresh state.
    """
    from cognee.modules.engine.operations.setup import setup

    await setup()


async def dataset_exists(patient_id: str) -> bool:
    await ensure_db_setup()
    datasets = await cognee.datasets.list_datasets()
    return any(d.name == patient_id for d in datasets)


async def graph_counts(patient_id: str = DEMO_PATIENT_ID) -> tuple[int, int] | None:
    """Best-effort node/edge count so we can sanity-check the graph built.

    With access control on, each dataset has its own graph database, so the
    dataset context must be entered first (same dance visualize_graph does).
    """
    try:
        from cognee.context_global_variables import set_database_global_context_variables
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.modules.data.methods import get_authorized_existing_datasets
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        datasets = await get_authorized_existing_datasets([patient_id], "read", user)
        if not datasets:
            return None
        async with set_database_global_context_variables(datasets[0].id, user.id):
            engine = await get_graph_engine()
            nodes, edges = await engine.get_graph_data()
            return len(nodes), len(edges)
    except Exception as exc:  # counting is a nice-to-have, never fail ingest
        print(f"  (could not count graph elements: {exc})")
        return None


async def export_graph_html(patient_id: str) -> None:
    try:
        # include_session_events=False keeps the visual to just the
        # document-derived knowledge graph — cleaner for judges.
        await cognee.visualize_graph(
            destination_file_path=str(graph_html_path(patient_id)),
            include_session_events=False,
            dataset=patient_id,
        )
        print(f"  Graph visualization written to {graph_html_path(patient_id)}")
    except Exception as exc:
        print(f"  (graph visualization failed, demo can still run: {exc})")


async def run_improve(patient_id: str) -> str:
    """Explicit lifecycle step 3: refine the freshly built graph.

    remember() is called with self_improvement=False so this improve() run
    is a single, visible, loggable step instead of an implicit background
    one — its output is surfaced in the terminal and the /ingest response.
    """
    started = time.monotonic()
    result = await cognee.improve(dataset=patient_id)
    elapsed = time.monotonic() - started
    summary = f"improve(dataset='{patient_id}') completed in {elapsed:.1f}s: {result!r}"
    print(f"  {summary}")
    return summary


async def remember_text(patient_id: str, text: str, filename: str,
                        digest: str | None = None) -> dict:
    """Ingest one piece of extracted text for a patient (upload path).

    `digest` is the content hash the caller deduped on (the raw uploaded
    bytes); it defaults to the text's own hash when not given.
    """
    await ensure_db_setup()
    registry.ensure_patient(patient_id)
    started = time.monotonic()
    await cognee.remember(
        text,
        dataset_name=patient_id,
        temporal_cognify=True,  # routed through to cognify() by remember()
        self_improvement=False,  # improve() is called explicitly instead
    )
    remember_seconds = round(time.monotonic() - started, 1)
    registry.record_document(
        patient_id, filename, digest or registry.content_hash(text.encode()))
    improve_log = await run_improve(patient_id)
    # No graph HTML export here: the UI hits /graph?refresh=1 right after an
    # upload, which re-exports on demand — exporting here doubled the wait.
    return {"ingested": filename, "improve": improve_log,
            "remember_seconds": remember_seconds}


async def run_ingestion(force: bool = False) -> dict:
    """Seed the demo patient from data/patient_records/ (idempotent)."""
    require_llm_key()
    await ensure_db_setup()

    if force:
        print("Force mode: pruning existing cognee data and system state...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        registry.forget_patient(DEMO_PATIENT_ID)
        await ensure_db_setup()

    registry.ensure_patient(DEMO_PATIENT_ID, DEMO_PATIENT_NAME)

    # Registry-based dedup: only remember() files whose content hash is new
    # for this patient. Safe to re-run any time.
    new_paths: list[str] = []
    for path_str in document_paths():
        from pathlib import Path

        digest = registry.hash_file(Path(path_str))
        if registry.document_exists(DEMO_PATIENT_ID, digest):
            continue
        new_paths.append(path_str)

    if not new_paths:
        print(f"All documents in {DOCUMENTS_DIR} already ingested for "
              f"'{DEMO_PATIENT_ID}' — nothing to do.")
        counts = await graph_counts()
        if counts:
            print(f"Existing graph: {counts[0]} nodes, {counts[1]} edges.")
        if not graph_html_path(DEMO_PATIENT_ID).exists():
            await export_graph_html(DEMO_PATIENT_ID)
        return {"skipped": True}

    print(f"remember(): ingesting {len(new_paths)} documents for patient "
          f"'{DEMO_PATIENT_ID}'...")
    for path in new_paths:
        print(f"  + {path.rsplit('/', 1)[-1]}")

    # Lifecycle verb 1: remember == add() + cognify() in permanent-memory
    # mode. temporal_cognify=True extracts dates/events into a time-aware
    # graph, enabling "what happened after X" queries.
    print("This makes LLM calls — expect a few minutes...")
    await cognee.remember(
        new_paths,
        dataset_name=DEMO_PATIENT_ID,
        temporal_cognify=True,
        self_improvement=False,
    )
    from pathlib import Path

    for path_str in new_paths:
        p = Path(path_str)
        registry.record_document(DEMO_PATIENT_ID, p.name, registry.hash_file(p))

    counts = await graph_counts()
    if counts:
        print(f"Knowledge graph built: {counts[0]} nodes, {counts[1]} edges.")
        if counts[0] < 10:
            print("  WARNING: suspiciously small graph — check the LLM output.")

    # Lifecycle verb 3: explicit, logged improve() pass.
    improve_log = await run_improve(DEMO_PATIENT_ID)

    await export_graph_html(DEMO_PATIENT_ID)
    print("Ingestion complete.")
    return {
        "skipped": False,
        "documents": len(new_paths),
        "nodes": counts[0] if counts else None,
        "edges": counts[1] if counts else None,
        "improve": improve_log,
    }


if __name__ == "__main__":
    asyncio.run(run_ingestion(force="--force" in sys.argv))
