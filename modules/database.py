"""
Local Database Module (SQLite)
-----------------------------------
Provides persistent storage for processed fingerprint records so the
system supports identification (matching a new print against a gallery)
and the analytics dashboard (pattern-type distribution, history, etc.).

SQLite was chosen because it requires no separate server process, ships
with Python's standard library (via the ``sqlite3`` module), and is
entirely appropriate for a single-user/local prototype such as this
coursework deliverable.
"""

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from modules.minutiae_detection import Minutia
from modules.singular_points import SingularPoint

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fingerprints.db")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT NOT NULL,
    subject_id TEXT,
    image_path TEXT NOT NULL,
    pattern_type TEXT,
    num_cores INTEGER,
    num_deltas INTEGER,
    num_minutiae INTEGER,
    estimated_dpi REAL,
    minutiae_json TEXT,
    singular_points_json TEXT,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    conn = get_connection()
    try:
        conn.execute(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _minutiae_to_json(minutiae: List[Minutia]) -> str:
    return json.dumps([asdict(m) for m in minutiae])


def _minutiae_from_json(raw: str) -> List[Minutia]:
    return [Minutia(**item) for item in json.loads(raw)] if raw else []


def _singular_to_json(points: List[SingularPoint]) -> str:
    return json.dumps([asdict(p) for p in points])


def _singular_from_json(raw: str) -> List[SingularPoint]:
    return [SingularPoint(**item) for item in json.loads(raw)] if raw else []


def insert_record(subject_name: str, subject_id: str, image_path: str,
                   pattern_type: str, minutiae: List[Minutia],
                   singular_points: List[SingularPoint], estimated_dpi: float) -> int:
    n_core = sum(1 for p in singular_points if p.kind == "core")
    n_delta = sum(1 for p in singular_points if p.kind == "delta")

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO fingerprints
               (subject_name, subject_id, image_path, pattern_type, num_cores,
                num_deltas, num_minutiae, estimated_dpi, minutiae_json,
                singular_points_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (subject_name, subject_id, image_path, pattern_type, n_core, n_delta,
             len(minutiae), estimated_dpi, _minutiae_to_json(minutiae),
             _singular_to_json(singular_points), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_records(hydrate: bool = True) -> List[dict]:
    """
    Returns every stored record as a plain dict. If ``hydrate`` is True,
    ``minutiae`` and ``singular_points`` are parsed back into dataclass
    lists (needed by the matcher); otherwise the raw JSON strings are
    kept as-is (cheaper, useful for just listing/browsing records).
    """
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM fingerprints ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()

    records = []
    for row in rows:
        record = dict(row)
        if hydrate:
            record["minutiae"] = _minutiae_from_json(record.pop("minutiae_json"))
            record["singular_points"] = _singular_from_json(record.pop("singular_points_json"))
        records.append(record)
    return records


def get_record(record_id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM fingerprints WHERE id = ?", (record_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    record = dict(row)
    record["minutiae"] = _minutiae_from_json(record.pop("minutiae_json"))
    record["singular_points"] = _singular_from_json(record.pop("singular_points_json"))
    return record


def delete_record(record_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM fingerprints WHERE id = ?", (record_id,))
        conn.commit()
    finally:
        conn.close()


def pattern_type_distribution() -> dict:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT pattern_type, COUNT(*) as cnt FROM fingerprints GROUP BY pattern_type"
        ).fetchall()
    finally:
        conn.close()
    return {row["pattern_type"]: row["cnt"] for row in rows}
