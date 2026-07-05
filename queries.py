"""Demo queries for the Patient Health Memory assistant"""

import asyncio
import re
import sys
from typing import Any

import cognee_setup  # must come before `import cognee` (sets env vars)
from cognee_setup import COGNEE_CLOUD_ENABLED, DEMO_PATIENT_ID, require_llm_key

import cognee
from cognee import SearchType

# ---------------------------------------------------------------------------
# The 5 demo questions.
#
# q1  SINGLE-HOP sanity check: fully answerable from document 1 alone.
#
# q2  MULTI-HOP: "atrial fibrillation" is named ONLY in document 1; the
#     November 2025 medication (apixaban) appears ONLY in document 4, which
#     refers back to the diagnosis merely as "the rhythm disorder documented
#     in the January 2025 discharge summary". Answering requires joining
#     doc 1 <-> doc 4 through the graph.
#
# q3  MULTI-HOP: the care chain Kulkarni -> Nair -> Menon is spread over
#     documents 1-5; no single document lists all three doctors' roles.
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
        "question": "What new medication was Anjali prescribed in November "
                    "2025, and what is the name of the condition it treats?",
    },
    {
        "id": "q3_multi_hop_care_chain",
        "label": "Multi-hop: care chain across five documents",
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

# Both sides of the comparison run with the SAME strict grounding prompt.
# Without it, the LLM answers "what is apixaban treating?" from general
# medical knowledge (apixaban -> AFib) without any document saying so —
# which is both an unfair baseline and exactly the hallucination a medical
# records product must not make.
GROUNDED_PROMPT = (
    "Answer using ONLY the provided context from this patient's records. "
    "Never use outside medical knowledge to fill gaps — if the context does "
    "not state a fact (for example the name of a condition), say the records "
    "you can see do not name it. Be concise."
)

# cognee appends an "Evidence:" block (chunk/document listing) to the
# completion when include_references=True. Split it off: text before is the
# clean answer, document names in it become the sources list.
_EVIDENCE_SPLIT = "\nEvidence:"
_DOC_NAME_RE = re.compile(r"of document (\S+)")


def split_answer_and_sources(text: str) -> tuple[str, list[str]]:
    answer, _, evidence = text.partition(_EVIDENCE_SPLIT)
    seen: set[str] = set()
    sources = [d for d in _DOC_NAME_RE.findall(evidence)
               if not (d in seen or seen.add(d))]
    return answer.strip(), sources


def unpack_search_results(results: list) -> tuple[str, list[str]]:
    """Flatten cognee search/recall results into (answer_text, source_names).

    Payload shapes vary by query type and API (str, list, dict, SearchResult,
    or pydantic Response*Entry from recall) — unpack defensively.
    """
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
        elif hasattr(item, "model_dump"):  # pydantic recall entries
            visit(item.model_dump())
        elif hasattr(item, "search_result"):
            visit(item.search_result)
        else:
            answers.append(str(item))

    for result in results:
        visit(result)

    joined = "\n".join(a.strip() for a in answers if a.strip())
    answer, evidence_sources = split_answer_and_sources(joined)
    seen: set[str] = set()
    all_sources = [s for s in sources + evidence_sources
                   if not (s in seen or seen.add(s))]
    return answer, all_sources


async def ask_graph(question: str, patient_id: str = DEMO_PATIENT_ID) -> tuple[str, list[str]]:
    """Graph answer via recall() (the product).

    GRAPH_COMPLETION_COT with max_iter=2: the chain-of-thought pass issues a
    follow-up retrieval that makes the cross-document hop (e.g. prescription
    -> discharge summary -> condition name) which single-shot triplet
    retrieval misses even at top_k=50 — verified empirically. Costs ~40-70s;
    the accuracy is the demo.
    """
    if COGNEE_CLOUD_ENABLED:
        # Single-shot GRAPH_COMPLETION on cloud: the hosted ingestion runs
        # its own refinement inside remember(), and we verified the fast
        # mode answers the multi-hop demo questions correctly there in
        # ~12s — chain-of-thought (~40-90s) is only needed for the local
        # graph, where single-shot provably misses the cross-document hop.
        from cloud import cloud_recall

        results = await cloud_recall(
            question, patient_id, "GRAPH_COMPLETION",
            top_k=30, system_prompt=GROUNDED_PROMPT, include_references=True)
    else:
        results = await cognee.recall(
            question,
            query_type=SearchType.GRAPH_COMPLETION_COT,
            datasets=[patient_id],
            top_k=30,
            retriever_specific_config={"max_iter": 2},
            system_prompt=GROUNDED_PROMPT,
            include_references=True,
        )
    return unpack_search_results(results)


async def ask_vector_baseline(question: str, patient_id: str = DEMO_PATIENT_ID) -> str:
    """Plain vector-retrieval RAG baseline — no graph traversal.

    top_k=3 chunks, the typical RAG-tutorial setup. With unlimited k on a
    small corpus the "baseline" would just read the whole archive into
    context, which neither scales nor resembles production vector RAG —
    the point of the comparison is what retrieval alone finds.
    """
    if COGNEE_CLOUD_ENABLED:
        from cloud import cloud_recall

        results = await cloud_recall(
            question, patient_id, "RAG_COMPLETION",
            top_k=3, system_prompt=GROUNDED_PROMPT)
    else:
        results = await cognee.recall(
            question,
            query_type=SearchType.RAG_COMPLETION,
            datasets=[patient_id],
            top_k=3,
            system_prompt=GROUNDED_PROMPT,
        )
    answer, _ = unpack_search_results(results)
    return answer


async def main(compare: bool) -> None:
    require_llm_key()
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'=' * 72}\n[{i}/5] {q['label']}\nQ: {q['question']}\n")
        answer, refs = await ask_graph(q["question"])
        print(f"GRAPH ANSWER (recall):\n{answer}")
        if refs:
            print(f"\nSources: {', '.join(refs)}")
        if compare:
            baseline = await ask_vector_baseline(q["question"])
            print(f"\nVECTOR-ONLY BASELINE (RAG_COMPLETION, top_k=3):\n{baseline}")
    print(f"\n{'=' * 72}\nDone.")


if __name__ == "__main__":
    asyncio.run(main(compare="--compare" in sys.argv))
