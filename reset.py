"""Reset script.

The per-patient path uses lifecycle verb 4: cognee.forget(dataset=patient_id)
— the same call the UI's "forget this patient" button triggers.
"""

import asyncio
import sys

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import graph_html_path

import cognee
import registry


async def forget_one(patient_id: str) -> dict:
    """Delete one patient's graph data + registry rows + exported graph."""
    result = await cognee.forget(dataset=patient_id)
    removed = registry.forget_patient(patient_id)
    html = graph_html_path(patient_id)
    if html.exists():
        html.unlink()
    print(f"forget(dataset='{patient_id}') -> {result!r}; "
          f"{removed} registry document rows removed.")
    return {"patient_id": patient_id, "forget_result": str(result),
            "registry_rows_removed": removed}


async def full_reset() -> None:
    print("Pruning cognee data (documents, chunks, graph, vectors)...")
    await cognee.prune.prune_data()
    print("Pruning cognee system state (metadata, datasets)...")
    await cognee.prune.prune_system(metadata=True)
    if registry.DB_PATH.exists():
        registry.DB_PATH.unlink()
        print("Removed patient registry (patients.db).")
    for html in graph_html_path("x").parent.glob("graph_*.html"):
        html.unlink()
    print("Reset complete. Run `uv run python ingest.py` to rebuild.")


if __name__ == "__main__":
    if "--patient" in sys.argv:
        pid = sys.argv[sys.argv.index("--patient") + 1]
        asyncio.run(forget_one(pid))
    else:
        asyncio.run(full_reset())
