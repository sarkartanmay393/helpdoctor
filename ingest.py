"""Ingestion pipeline: for now load the 5 demo synthetic patient documents into cognee."""

import asyncio
import sys

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import DATASET_NAME, GRAPH_HTML_PATH, document_paths, require_llm_key

import cognee


async def ensure_db_setup() -> None:
    """Create cognee's relational schema on first run (idempotent).

    Without this, any cognee call that resolves the default user (including
    datasets.list_datasets) raises DatabaseNotCreatedError on a fresh state.
    """
    from cognee.modules.engine.operations.setup import setup

    await setup()


async def dataset_exists() -> bool:
    await ensure_db_setup()
    datasets = await cognee.datasets.list_datasets()
    return any(d.name == DATASET_NAME for d in datasets)


async def graph_counts() -> tuple[int, int] | None:
    """Best-effort node/edge count so we can sanity-check the graph built."""
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine

        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
        return len(nodes), len(edges)
    except Exception as exc:  # counting is a nice-to-have, never fail ingest
        print(f"  (could not count graph elements: {exc})")
        return None


async def export_graph_html() -> None:
    try:
        # include_session_events=False keeps the visual to just the
        # document-derived knowledge graph — cleaner for judges.
        await cognee.visualize_graph(
            destination_file_path=str(GRAPH_HTML_PATH),
            include_session_events=False,
            dataset=DATASET_NAME,
        )
        print(f"  Graph visualization written to {GRAPH_HTML_PATH}")
    except Exception as exc:
        print(f"  (graph visualization failed, demo can still run: {exc})")


async def run_ingestion(force: bool = False) -> dict:
    require_llm_key()

    if force:
        print("Force mode: pruning existing cognee data and system state...")
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    elif await dataset_exists():
        print(f"Dataset '{DATASET_NAME}' already ingested — skipping.")
        print("Re-run with --force to wipe and rebuild.")
        counts = await graph_counts()
        if counts:
            print(f"Existing graph: {counts[0]} nodes, {counts[1]} edges.")
        if not GRAPH_HTML_PATH.exists():
            await export_graph_html()
        return {"skipped": True}

    docs = document_paths()
    print(f"Adding {len(docs)} documents to dataset '{DATASET_NAME}'...")
    for path in docs:
        print(f"  + {path.rsplit('/', 1)[-1]}")
    await cognee.add(docs, dataset_name=DATASET_NAME)

    # temporal_cognify=True (supported in cognee 1.2.2) extracts dates and
    # events into a time-aware graph, enabling "what happened after X" queries.
    print("Running cognify with temporal extraction (this makes LLM calls, "
          "expect a few minutes)...")
    await cognee.cognify(datasets=[DATASET_NAME], temporal_cognify=True)

    counts = await graph_counts()
    if counts:
        print(f"Knowledge graph built: {counts[0]} nodes, {counts[1]} edges.")
        if counts[0] < 10:
            print("  WARNING: suspiciously small graph — check the LLM output.")
    else:
        print("Knowledge graph built (element counting unavailable).")

    await export_graph_html()
    print("Ingestion complete.")
    return {
        "skipped": False,
        "documents": len(docs),
        "nodes": counts[0] if counts else None,
        "edges": counts[1] if counts else None,
    }


if __name__ == "__main__":
    asyncio.run(run_ingestion(force="--force" in sys.argv))
