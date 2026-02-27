from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Dict, Any, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "corpus"
DB_PATH = DATA_DIR / "corpus.db"
REGISTRY_JSONL = DATA_DIR / "page_registry.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS page_registry (
    page_id TEXT PRIMARY KEY,
    pdf_name TEXT,
    page_number TEXT,
    has_akkadian INTEGER,
    lang_hint TEXT,
    ocr_quality_hint TEXT,
    text_norm TEXT,
    text_raw TEXT
);
CREATE TABLE IF NOT EXISTS philology (
    page_id TEXT PRIMARY KEY,
    json TEXT
);
CREATE TABLE IF NOT EXISTS social (
    page_id TEXT PRIMARY KEY,
    json TEXT
);
"""


def ensure_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


# Alias used by extractors
init_db = ensure_db


@contextmanager
def _maybe_conn(db_path: Path, conn: Optional[sqlite3.Connection] = None) -> Iterator[sqlite3.Connection]:
    """
    Provide a connection, reusing the provided one or opening a temporary.
    """
    if conn is not None:
        yield conn
    else:
        tmp_conn = ensure_db(db_path)
        try:
            yield tmp_conn
        finally:
            tmp_conn.close()


def upsert_pages(conn: sqlite3.Connection, records: Iterable[Dict[str, Any]]) -> None:
    rows = [
        (
            rec["page_id"],
            rec.get("pdf_name", ""),
            str(rec.get("page_number", "")),
            int(bool(rec.get("has_akkadian", False))),
            rec.get("lang_hint", ""),
            rec.get("ocr_quality_hint", ""),
            rec.get("text_norm", ""),
            rec.get("text_raw", ""),
        )
        for rec in records
    ]
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO page_registry
            (page_id, pdf_name, page_number, has_akkadian, lang_hint, ocr_quality_hint, text_norm, text_raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def fetch_pages_batch(
    db_path: Path = DB_PATH,
    limit: int = 1000,
    offset: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    with _maybe_conn(db_path, conn) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(
            """
            SELECT page_id, pdf_name, page_number, lang_hint, text_norm, text_raw
            FROM page_registry
            ORDER BY page_id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def upsert_payload(
    db_path: Path,
    table: str,
    page_id: str,
    payload: Any,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    if table not in {"philology", "social"}:
        raise ValueError(f"Unsupported table: {table}")
    serialized = json.dumps(payload, ensure_ascii=False)
    with _maybe_conn(db_path, conn) as c:
        if conn is None:
            with c:
                c.execute(
                    f"INSERT OR REPLACE INTO {table} (page_id, json) VALUES (?, ?)",
                    (page_id, serialized),
                )
        else:
            c.execute(
                f"INSERT OR REPLACE INTO {table} (page_id, json) VALUES (?, ?)",
                (page_id, serialized),
            )


def upsert_json(conn: sqlite3.Connection, table: str, records: Dict[str, Any]) -> None:
    rows = [(pid, json.dumps(payload, ensure_ascii=False)) for pid, payload in records.items()]
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} (page_id, json) VALUES (?, ?)",
            rows,
        )


def load_registry_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_registry_into_db(
    registry_path: Path = REGISTRY_JSONL,
    db_path: Path = DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> int:
    """
    Load page_registry.jsonl into SQLite (corpus.db).
    Returns number of rows written.
    """
    if conn is None:
        conn = ensure_db(db_path)
    records = load_registry_jsonl(registry_path)
    upsert_pages(conn, records)
    return len(records)


if __name__ == "__main__":
    conn = ensure_db()
    print(f"Initialized DB at {DB_PATH}")
    if REGISTRY_JSONL.exists():
        count = load_registry_into_db(REGISTRY_JSONL, DB_PATH, conn)
        print(f"Loaded {count} page records from {REGISTRY_JSONL.name} into {DB_PATH.name}")
    else:
        print(f"Registry not found: {REGISTRY_JSONL}")
