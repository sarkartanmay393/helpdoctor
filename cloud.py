"""Cognee Cloud mode: the same four lifecycle verbs, executed on a hosted
Cognee instance instead of the local in-process engine.

Enabled with COGNEE_CLOUD=true (see cognee_setup.py for the connection
settings). Uses cognee's own CloudClient (X-Api-Key auth against the
/api/v1/remember|recall|improve|forget REST surface), so the call pattern
mirrors local mode one-to-one. The patient registry (patients.db) stays
local in both modes — it is a lookup index, not the record store.

Differences from local mode, by design:
  - The graph visualization panel is local-mode only (the HTML export
    renders from the local graph database).
  - recall() runs with the server's default chain-of-thought settings
    (the hosted API doesn't accept retriever_specific_config).
"""

from cognee_setup import COGNEE_CLOUD_KEY, COGNEE_CLOUD_URL

_client = None


def get_client():
    global _client
    if _client is None:
        from cognee.api.v1.serve.cloud_client import CloudClient

        _client = CloudClient(COGNEE_CLOUD_URL, COGNEE_CLOUD_KEY)
    return _client


async def cloud_health() -> bool:
    return await get_client()._health_check()


async def cloud_remember_text(patient_id: str, text: str) -> dict:
    return await get_client().remember(text, dataset_name=patient_id)


async def cloud_recall(question: str, patient_id: str, query_type: str,
                       top_k: int, system_prompt: str,
                       include_references: bool = False) -> list:
    # scope="graph": restrict to the dataset's knowledge graph. Without it
    # the hosted side merges its session-memory cache (per API user, NOT per
    # dataset) into results — we observed an answer from an already-forgotten
    # dataset resurfacing through that cache.
    return await get_client().recall(
        question,
        query_type=query_type,
        datasets=[patient_id],
        top_k=top_k,
        system_prompt=system_prompt,
        include_references=include_references,
        scope="graph",
    )


async def cloud_improve(patient_id: str) -> dict:
    return await get_client().improve(dataset=patient_id)


async def cloud_forget(patient_id: str) -> dict:
    return await get_client().forget(dataset=patient_id)


async def cloud_dataset_exists(patient_id: str) -> bool:
    session = await get_client()._get_session()
    async with session.get(f"{get_client().service_url}/api/v1/datasets") as resp:
        if resp.status >= 400:
            raise RuntimeError(f"Remote datasets list failed ({resp.status}): "
                               f"{await resp.text()}")
        datasets = await resp.json()
    for d in datasets if isinstance(datasets, list) else []:
        name = d.get("name") if isinstance(d, dict) else None
        if name == patient_id:
            return True
    return False
