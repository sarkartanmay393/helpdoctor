"""Reset script: wipe all cognee data + system state for a clean re-run."""

import asyncio

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import GRAPH_HTML_PATH

import cognee


async def main() -> None:
    print("Pruning cognee data (documents, chunks, graph, vectors)...")
    await cognee.prune.prune_data()
    print("Pruning cognee system state (metadata, datasets)...")
    await cognee.prune.prune_system(metadata=True)
    if GRAPH_HTML_PATH.exists():
        GRAPH_HTML_PATH.unlink()
        print("Removed exported graph HTML.")
    print("Reset complete. Run `uv run python ingest.py` to rebuild.")


if __name__ == "__main__":
    asyncio.run(main())
