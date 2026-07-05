"""Doctor's Desk: deterministic question router for the global doctor chat.

Routing (no LLM involved except the final recall path):
  1. Fuzzy-match a patient out of the question text against the registry
     (difflib.get_close_matches — stdlib, no ML).
  2. Keyword rules classify the question:
       schedule/appointment (or "today" with no patient) -> seed schedule
       matched patient + documents/records/how many       -> registry SQL
       matched patient + "last visit"/"when...visit/seen" -> registry SQL
       matched patient + anything else                    -> recall() (caller)
       no patient, not schedule                           -> "not identified"
Out of scope by design: booking, calendars, availability, and any
cross-patient aggregate reasoning ("which patients have X").
"""

import difflib
import re
from datetime import date

import registry

# ---------------------------------------------------------------------------
# PLACEHOLDER DEMO DATA — this is NOT a real scheduling system. A hardcoded
# "today" roster so the Doctor's Desk has something to answer schedule
# questions from. patient_ids reference real rows in the SQLite registry.
# ---------------------------------------------------------------------------
def today_schedule() -> list[dict]:
    today = date.today().isoformat()
    return [
        {"date": today, "time": "09:30", "patient_id": "anjali_deshpande",
         "patient_name": "Anjali Deshpande",
         "reason": "Echocardiogram results review (rhythm follow-up)"},
        {"date": today, "time": "11:00", "patient_id": "ravi_kumar",
         "patient_name": "Ravi Kumar",
         "reason": "Allergic rhinitis — response to cetirizine"},
        {"date": today, "time": "15:15", "patient_id": "anjali_deshpande",
         "patient_name": "Anjali Deshpande",
         "reason": "Teleconsult: kidney function test report"},
    ]


# Words in registry display names that aren't part of a person's name.
_NAME_NOISE = {"synthetic", "demo", "patient", "test"}

_SCHEDULE_WORDS = ("schedule", "appointment", "appointments")
_DOC_COUNT_WORDS = ("how many", "document", "documents", "record", "records", "files")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def match_patient(question: str) -> dict | None:
    """Fuzzy-match one registry patient out of free text; None if ambiguous
    or absent. Matches on name/id tokens so 'Anjali', 'Deshpande' or
    'anjali_deshpande' all resolve."""
    q_tokens = _tokens(question)
    best, best_score = None, 0
    for p in registry.list_patients():
        name_tokens = [t for t in _tokens(f"{p['name'] or ''} {p['patient_id']}")
                       if t not in _NAME_NOISE]
        # cutoff 0.8 tolerates one-letter misspellings ("Anjaly" -> "anjali",
        # ratio 0.83) while "ravi" vs "rani" (0.75) still stays excluded.
        score = sum(1 for qt in q_tokens
                    if difflib.get_close_matches(qt, name_tokens, n=1, cutoff=0.8))
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 1 else None


def _fmt_when(iso: str) -> str:
    return (iso or "?")[:16].replace("T", " ")


def route_question(question: str) -> dict:
    """Returns {"route", "answer", "patient_id"}. route == "recall" means the
    caller should run recall() scoped to patient_id — everything else is
    answered here without an LLM."""
    q = question.lower()
    patient = match_patient(question)
    pid = patient["patient_id"] if patient else None
    display = (patient["name"] or pid) if patient else None

    # Schedule questions read the placeholder roster. A bare "today" only
    # counts as a schedule cue when no patient was named, so "how is Anjali
    # today" still routes to her records.
    if any(w in q for w in _SCHEDULE_WORDS) or ("today" in q and not patient):
        entries = today_schedule()
        if patient:
            entries = [e for e in entries if e["patient_id"] == pid]
        if not entries:
            answer = "Nothing on today's schedule" + (f" for {display}." if patient else ".")
        else:
            lines = [f"{e['time']} — {e['patient_name']}: {e['reason']}" for e in entries]
            answer = f"Today's schedule ({entries[0]['date']}):\n" + "\n".join(lines)
        return {"route": "seed_data", "answer": answer, "patient_id": pid}

    if patient:
        docs = registry.list_documents(pid)

        if "last visit" in q or ("when" in q and ("visit" in q or "seen" in q)):
            if not docs:
                answer = f"No records have been ingested for {display} yet."
            else:
                latest = max(docs, key=lambda d: d["ingested_at"] or "")
                answer = (f"{display}'s most recent record was added on "
                          f"{_fmt_when(latest['ingested_at'])} "
                          f"({latest['filename']}). Dates reflect when records "
                          f"entered memory, not clinical visit dates.")
            return {"route": "sql", "answer": answer, "patient_id": pid}

        if any(w in q for w in _DOC_COUNT_WORDS):
            if not docs:
                answer = f"No documents on file for {display}."
            else:
                names = "\n".join(f"- {d['filename']}" for d in docs)
                answer = f"{display} has {len(docs)} documents on file:\n{names}"
            return {"route": "sql", "answer": answer, "patient_id": pid}

        # Clinical/free-form question about a named patient -> graph recall,
        # scoped to that patient's memory. The caller runs the LLM part.
        return {"route": "recall", "answer": None, "patient_id": pid}

    return {"route": "not_identified",
            "answer": "I couldn't identify a patient in that question. "
                      "Try including the patient's name.",
            "patient_id": None}
