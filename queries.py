"""Demo queries for the Patient Health Memory assistant"""

import asyncio
import sys
from typing import Any

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import DATASET_NAME, require_llm_key

import cognee
from cognee import SearchType

# ---------------------------------------------------------------------------
# The 5 demo questions.
#
# q1  SINGLE-HOP sanity check: fully answerable from document 1 alone.
#
# q2  MULTI-HOP: "atrial fibrillation" is named ONLY in document 1; the
#     medication treating it (apixaban) appears ONLY in document 4, which
#     refers back to the diagnosis merely as "the rhythm disorder documented
#     in the January 2025 discharge summary". Answering requires joining
#     doc 1 <-> doc 4 through the graph.
#
# q3  MULTI-HOP: the care chain Kulkarni -> Nair -> Menon is spread over
#     documents 1, 2, 3, 4 and 5; no single document lists all three doctors'
#     roles in sequence. (Doc 5 comes closest but relies on the others for
#     what each doctor actually did.)
#
# q4  TEMPORAL: requires ordering events in time — symptoms before the
#     November 2025 consultation (docs 3/4) vs. after it (doc 5) — which is
#     what temporal_cognify=True during ingestion enables.
#
# q5  THE KEYWORD-SEARCH KILLER (narrate this one in the pitch): a keyword or
#     pure vector search for "apixaban" retrieves ONLY document 4 — and
#     document 4 deliberately never names the condition. A naive RAG answer
#     is at best "stroke prevention for a rhythm disorder". The graph
#     traverses: apixaban -> prescribed for -> rhythm disorder documented in
#     Jan 2025 discharge summary -> atrial fibrillation (doc 1), and can name
#     the actual condition.
# ---------------------------------------------------------------------------
DEMO_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "q1_single_hop",
        "label": "Single-hop sanity check (one document)",
        "question": "Why was Anjali Deshpande hospitalized in January 2025, "
                    "and which doctor treated her there?",
    },
    {
        "id": "q2_multi_hop_medication",
        "label": "Multi-hop: medication in doc 4, condition named only in doc 1",
        "question": "What medication was prescribed for the heart condition "
                    "that Dr. Kulkarni diagnosed during the January 2025 "
                    "hospitalization, and who prescribed it?",
    },
    {
        "id": "q3_multi_hop_care_chain",
        "label": "Multi-hop: care chain across all five documents",
        "question": "Which doctors have been involved in Anjali Deshpande's "
                    "cardiac care, and how did she get from one to the next?",
    },
    {
        "id": "q4_temporal",
        "label": "Temporal: before vs. after the specialist visit",
        "question": "How did Anjali Deshpande's heart condition change after "
                    "she started treatment with Dr. Menon in November 2025?",
    },
    {
        "id": "q5_keyword_killer",
        "label": "Keyword-search killer: doc 4 never names the condition",
        "question": "What medical condition is the apixaban prescribed by "
                    "Dr. Menon actually treating?",
    },
]


def unpack_search_results(results: list) -> tuple[str, list[str]]:
    """Flatten cognee SearchResult objects into (answer_text, source_names)"""
    
    answers: list[str] = []
    sources: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            answers.append(item)
        elif isinstance(item, dict):
            text = item.get("completion") or item.get("answer") or item.get("text")
            if isinstance(text, str):
                answers.append(text)
            elif text is not None:
                visit(text)
            for ref in item.get("references") or item.get("context") or []:
                if isinstance(ref, dict):
                    name = (ref.get("document_name") or ref.get("name")
                            or ref.get("title") or ref.get("id"))
                    if name:
                        sources.append(str(name))
                elif isinstance(ref, str):
                    sources.append(ref)
        elif isinstance(item, list):
            for sub in item:
                visit(sub)
        else:
            answers.append(str(item))

    for result in results:
        visit(getattr(result, "search_result", result))

    seen: set[str] = set()
    unique_sources = [s for s in sources if not (s in seen or seen.add(s))]
    return "\n".join(a.strip() for a in answers if a.strip()), unique_sources


async def ask_graph(question: str) -> tuple[str, list[str]]:
    """Graph-grounded answer (the product). include_references=True asks
    cognee to attach source attribution when the retriever supports it."""
    results = await cognee.search(
        query_text=question,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
        include_references=True,
    )
    return unpack_search_results(results)


async def ask_vector_baseline(question: str) -> str:
    """Plain vector-retrieval RAG over chunks — the naive baseline that the
    demo contrasts against. No graph traversal happens here."""
    results = await cognee.search(
        query_text=question,
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET_NAME],
    )
    answer, _ = unpack_search_results(results)
    return answer


async def main(compare: bool) -> None:
    require_llm_key()
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'=' * 72}\n[{i}/5] {q['label']}\nQ: {q['question']}\n")
        answer, refs = await ask_graph(q["question"])
        print(f"GRAPH ANSWER:\n{answer}")
        if refs:
            print(f"\nSources: {', '.join(refs)}")
        if compare:
            baseline = await ask_vector_baseline(q["question"])
            print(f"\nVECTOR-ONLY BASELINE (RAG_COMPLETION):\n{baseline}")
    print(f"\n{'=' * 72}\nDone.")


if __name__ == "__main__":
    asyncio.run(main(compare="--compare" in sys.argv))
