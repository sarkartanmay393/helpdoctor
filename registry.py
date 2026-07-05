"""Patient registry: a tiny lookup index (SQLite locally, MySQL when hosted).

IMPORTANT: this database is a LOOKUP INDEX ONLY — patient ids/names and
which files were already ingested (by content hash, for dedup). It stores
no medical content. If this registry and Cognee's data ever disagree,
Cognee is the source of truth for content; this table is never a fallback
record store.

Document rows are scoped by ingestion MODE ("local" or "cloud"): the local
graph and a Cognee Cloud tenant are separate stores, so "already ingested"
must be answered per store. Patients (ids/names) are shared across modes.

Backends:
  - SQLite (patients.db) — default; always used in local mode.
  - MySQL (AWS RDS) — used in cloud mode when DB_HOST/DB_USER/DB_PASSWORD/
    DB_DATABASE are set, so a hosted deployment keeps its registry across
    redeploys (container disks are ephemeral). If MySQL is configured but
    unreachable (e.g. the RDS security group only allows the host's IP,
    not a dev laptop), we warn once and fall back to SQLite rather than
    hanging every request.
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cognee_setup import COGNEE_CLOUD_ENABLED, PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "patients.db"

# The active ingestion mode for this process; every document row is tagged
# with the mode it was ingested under.
MODE = "cloud" if COGNEE_CLOUD_ENABLED else "local"

_MYSQL_CONFIGURED = COGNEE_CLOUD_ENABLED and all(
    os.environ.get(v) for v in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_DATABASE"))

# None = undecided (first use probes MySQL), True/False after that.
_mysql_available: bool | None = None if _MYSQL_CONFIGURED else False

_SQLITE_SCHEMA = """
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

_MYSQL_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS patients (
        patient_id VARCHAR(128) PRIMARY KEY,
        name VARCHAR(255),
        created_at VARCHAR(64)
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        id INT AUTO_INCREMENT PRIMARY KEY,
        patient_id VARCHAR(128),
        filename VARCHAR(512),
        content_hash VARCHAR(64),
        ingested_at VARCHAR(64),
        mode VARCHAR(16) DEFAULT 'local',
        INDEX idx_docs_patient (patient_id),
        INDEX idx_docs_hash (content_hash)
    )""",
]


def _mysql_connect():
    import mysql.connector

    ssl_ca = os.environ.get("DB_SSL_CA", str(PROJECT_ROOT / "global-bundle.pem"))
    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", "3306")),
        database=os.environ["DB_DATABASE"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        ssl_disabled=False,
        ssl_ca=ssl_ca if Path(ssl_ca).exists() else None,
        autocommit=True,
        connection_timeout=8,
    )


def _probe_mysql() -> bool:
    global _mysql_available
    if _mysql_available is not None:
        return _mysql_available
    try:
        conn = _mysql_connect()
        cur = conn.cursor()
        for stmt in _MYSQL_SCHEMA:
            cur.execute(stmt)
        cur.close()
        conn.close()
        _mysql_available = True
        print("registry: using MySQL backend (persistent across redeploys)")
    except Exception as exc:
        _mysql_available = False
        print(f"registry: MySQL configured but unreachable ({exc}) — "
              f"falling back to SQLite (patients.db)")
    return _mysql_available


def _sqlite_migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(_SQLITE_SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    if "mode" not in cols:
        # Everything ingested before this column existed was local mode.
        conn.execute("ALTER TABLE documents ADD COLUMN mode TEXT DEFAULT 'local'")


@contextmanager
def _conn():
    """Yields (connection, paramstyle_char). Caller uses '?' placeholders;
    they are rewritten for MySQL."""
    if _probe_mysql():
        conn = _mysql_connect()
        try:
            yield conn, "%s"
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            _sqlite_migrate(conn)
            yield conn, "?"
            conn.commit()
        finally:
            conn.close()


def _rows(cur, is_mysql: bool) -> list[dict]:
    if is_mysql:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    return [dict(r) for r in cur.fetchall()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(data: bytes) -> str:
    # Dedup fingerprint only — not a security boundary.
    return hashlib.sha256(data).hexdigest()


def ensure_patient(patient_id: str, name: str | None = None) -> None:
    with _conn() as (conn, p):
        cur = conn.cursor()
        if p == "%s":  # MySQL: portable two-step upsert
            cur.execute(
                "INSERT IGNORE INTO patients (patient_id, name, created_at) "
                "VALUES (%s, %s, %s)", (patient_id, name, _now()))
            if name:
                cur.execute("UPDATE patients SET name = %s WHERE patient_id = %s",
                            (name, patient_id))
        else:
            cur.execute(
                "INSERT INTO patients (patient_id, name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(patient_id) DO UPDATE SET "
                "name = COALESCE(excluded.name, name)",
                (patient_id, name, _now()))
        cur.close()


def list_patients() -> list[dict]:
    with _conn() as (conn, p):
        cur = conn.cursor()
        cur.execute("SELECT patient_id, name, created_at FROM patients "
                    "ORDER BY created_at")
        rows = _rows(cur, p == "%s")
        cur.close()
    return rows


def list_documents(patient_id: str) -> list[dict]:
    """Documents ingested for this patient in the ACTIVE mode's store."""
    with _conn() as (conn, p):
        cur = conn.cursor()
        cur.execute(
            f"SELECT filename, content_hash, ingested_at FROM documents "
            f"WHERE patient_id = {p} AND mode = {p} ORDER BY ingested_at",
            (patient_id, MODE))
        rows = _rows(cur, p == "%s")
        cur.close()
    return rows


def document_exists(patient_id: str, digest: str) -> bool:
    with _conn() as (conn, p):
        cur = conn.cursor()
        cur.execute(
            f"SELECT 1 FROM documents WHERE patient_id = {p} "
            f"AND content_hash = {p} AND mode = {p}",
            (patient_id, digest, MODE))
        row = cur.fetchone()
        cur.close()
    return row is not None


def record_document(patient_id: str, filename: str, digest: str) -> None:
    with _conn() as (conn, p):
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO documents (patient_id, filename, content_hash, "
            f"ingested_at, mode) VALUES ({p}, {p}, {p}, {p}, {p})",
            (patient_id, filename, digest, _now(), MODE))
        cur.close()


def forget_patient(patient_id: str) -> int:
    """Remove this patient's registry rows for the ACTIVE mode (the graph
    deletion is cognee.forget, which also acts on the active store). The
    patient id itself is only removed once no mode has documents left."""
    with _conn() as (conn, p):
        cur = conn.cursor()
        cur.execute(f"DELETE FROM documents WHERE patient_id = {p} AND mode = {p}",
                    (patient_id, MODE))
        removed = cur.rowcount
        cur.execute(f"SELECT COUNT(*) FROM documents WHERE patient_id = {p}",
                    (patient_id,))
        remaining = cur.fetchone()[0]
        if remaining == 0:
            cur.execute(f"DELETE FROM patients WHERE patient_id = {p}",
                        (patient_id,))
        cur.close()
        return removed


def hash_file(path: Path) -> str:
    return content_hash(path.read_bytes())
