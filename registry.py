"""Patient registry: a tiny SQLite lookup index (stdlib sqlite3).

IMPORTANT: this database is a LOOKUP INDEX ONLY — patient ids/names and
which files were already ingested (by content hash, for dedup). It stores
no medical content. If this registry and Cognee's data ever disagree,
Cognee is the source of truth for content; this table is never a fallback
record store.
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cognee_setup import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "patients.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    patient_id TEXT,
    filename TEXT,
    content_hash TEXT,
    ingested_at TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(data: bytes) -> str:
    # Dedup fingerprint only — not a security boundary.
    return hashlib.sha256(data).hexdigest()


def ensure_patient(patient_id: str, name: str | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO patients (patient_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(patient_id) DO UPDATE SET name = COALESCE(excluded.name, name)",
            (patient_id, name, _now()),
        )


def list_patients() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT patient_id, name, created_at FROM patients ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def list_documents(patient_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT filename, content_hash, ingested_at FROM documents "
            "WHERE patient_id = ? ORDER BY ingested_at",
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def document_exists(patient_id: str, digest: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM documents WHERE patient_id = ? AND content_hash = ?",
            (patient_id, digest),
        ).fetchone()
    return row is not None


def record_document(patient_id: str, filename: str, digest: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (patient_id, filename, content_hash, ingested_at) "
            "VALUES (?, ?, ?, ?)",
            (patient_id, filename, digest, _now()),
        )


def forget_patient(patient_id: str) -> int:
    """Remove a patient's registry rows (the graph deletion is cognee.forget)."""
    with _conn() as c:
        cur = c.execute("DELETE FROM documents WHERE patient_id = ?", (patient_id,))
        c.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        return cur.rowcount


def hash_file(path: Path) -> str:
    return content_hash(path.read_bytes())
