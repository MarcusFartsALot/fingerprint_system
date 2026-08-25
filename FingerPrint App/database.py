"""SQLite persistence for students, fingerprint templates and attendance."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
REFERENCE_DIR = DATA_DIR / "references"
DATABASE_PATH = DATA_DIR / "attendance.db"


def local_now() -> datetime:
    return datetime.now().astimezone()


@contextmanager
def connection(database_path: str | Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(database_path, timeout=20)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    database.execute("PRAGMA journal_mode = WAL")
    try:
        yield database
        database.commit()
    except Exception:
        database.rollback()
        raise
    finally:
        database.close()


def initialise_database(database_path: str | Path = DATABASE_PATH) -> None:
    """Create the local schema idempotently."""

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with connection(database_path) as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                programme TEXT NOT NULL,
                study_year INTEGER NOT NULL CHECK(study_year BETWEEN 1 AND 8),
                tutorial_group TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'Active',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fingerprint_templates (
                template_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                finger_label TEXT NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                reference_path TEXT,
                quality REAL NOT NULL,
                clarity REAL NOT NULL,
                minutiae_count INTEGER NOT NULL,
                profile_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS class_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                course_name TEXT NOT NULL,
                venue TEXT NOT NULL,
                lecturer TEXT NOT NULL,
                starts_at TEXT NOT NULL,
                grace_minutes INTEGER NOT NULL DEFAULT 15,
                ends_at TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES class_sessions(session_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                marked_at TEXT NOT NULL,
                attendance_status TEXT NOT NULL CHECK(attendance_status IN ('Present', 'Late')),
                similarity REAL NOT NULL,
                capture_quality REAL NOT NULL,
                processing_ms REAL NOT NULL,
                UNIQUE(session_id, student_id)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_templates_student
                ON fingerprint_templates(student_id);
            CREATE INDEX IF NOT EXISTS idx_attendance_session
                ON attendance(session_id);
            CREATE INDEX IF NOT EXISTS idx_attendance_marked
                ON attendance(marked_at);
            """
        )
        # Non-destructive migration for databases created by version 1.0.
        columns = {
            row["name"]
            for row in database.execute("PRAGMA table_info(fingerprint_templates)").fetchall()
        }
        if "reference_path" not in columns:
            database.execute("ALTER TABLE fingerprint_templates ADD COLUMN reference_path TEXT")
        if "profile_json" not in columns:
            database.execute(
                "ALTER TABLE fingerprint_templates ADD COLUMN profile_json TEXT NOT NULL DEFAULT '{}'"
            )


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def add_audit_event(
    event_type: str,
    details: dict[str, Any] | str,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    payload = details if isinstance(details, str) else json.dumps(details, ensure_ascii=True)
    with connection(database_path) as database:
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            (event_type, payload, local_now().isoformat()),
        )


def enrol_student(
    student: dict[str, Any],
    templates: list[dict[str, Any]],
    database_path: str | Path = DATABASE_PATH,
) -> None:
    """Insert student metadata and one or more processed fingerprint templates."""

    created_at = local_now().isoformat()
    with connection(database_path) as database:
        database.execute(
            """
            INSERT INTO students(
                student_id, full_name, programme, study_year, tutorial_group,
                email, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'Active', ?)
            """,
            (
                student["student_id"].strip().upper(),
                student["full_name"].strip(),
                student["programme"].strip(),
                int(student["study_year"]),
                student["tutorial_group"].strip(),
                student.get("email", "").strip(),
                created_at,
            ),
        )
        for template in templates:
            database.execute(
                """
                INSERT INTO fingerprint_templates(
                    student_id, finger_label, image_path, reference_path,
                    quality, clarity, minutiae_count, profile_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student["student_id"].strip().upper(),
                    template["finger_label"],
                    str(template["image_path"]),
                    str(template["reference_path"]) if template.get("reference_path") else None,
                    float(template["quality"]),
                    float(template["clarity"]),
                    int(template["minutiae_count"]),
                    json.dumps(template.get("profile", {}), ensure_ascii=True),
                    created_at,
                ),
            )
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            (
                "student_enrolled",
                json.dumps(
                    {
                        "student_id": student["student_id"].strip().upper(),
                        "templates": len(templates),
                    }
                ),
                created_at,
            ),
        )


def add_fingerprint_templates(
    student_id: str,
    templates: list[dict[str, Any]],
    database_path: str | Path = DATABASE_PATH,
) -> None:
    """Add current-format captures to an existing student without deleting history."""

    canonical_id = student_id.strip().upper()
    if not templates:
        raise ValueError("At least one fingerprint template is required.")
    created_at = local_now().isoformat()
    with connection(database_path) as database:
        exists = database.execute(
            "SELECT 1 FROM students WHERE student_id = ?",
            (canonical_id,),
        ).fetchone()
        if not exists:
            raise ValueError(f"Student {canonical_id} does not exist.")
        for template in templates:
            database.execute(
                """
                INSERT INTO fingerprint_templates(
                    student_id, finger_label, image_path, reference_path,
                    quality, clarity, minutiae_count, profile_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_id,
                    template["finger_label"],
                    str(template["image_path"]),
                    str(template["reference_path"]) if template.get("reference_path") else None,
                    float(template["quality"]),
                    float(template["clarity"]),
                    int(template.get("minutiae_count", 0)),
                    json.dumps(template.get("profile", {}), ensure_ascii=True),
                    created_at,
                ),
            )
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            (
                "fingerprint_templates_added",
                json.dumps({"student_id": canonical_id, "templates": len(templates)}),
                created_at,
            ),
        )


def get_student(student_id: str, database_path: str | Path = DATABASE_PATH) -> dict[str, Any] | None:
    with connection(database_path) as database:
        row = database.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id.strip().upper(),)
        ).fetchone()
    return dict(row) if row else None


def list_students(database_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with connection(database_path) as database:
        return _rows(
            database.execute(
                """
                SELECT s.student_id, s.full_name, s.programme, s.study_year,
                       s.tutorial_group, s.email, s.status, s.created_at,
                       COUNT(t.template_id) AS templates,
                       ROUND(AVG(t.quality), 3) AS average_quality
                FROM students s
                LEFT JOIN fingerprint_templates t ON t.student_id = s.student_id
                GROUP BY s.student_id
                ORDER BY s.full_name COLLATE NOCASE
                """
            )
        )


def list_templates(database_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with connection(database_path) as database:
        return _rows(
            database.execute(
                """
                SELECT t.*, s.full_name, s.programme, s.study_year, s.tutorial_group,
                       s.email, s.status AS student_status
                FROM fingerprint_templates t
                JOIN students s ON s.student_id = t.student_id
                WHERE s.status = 'Active'
                ORDER BY s.student_id, t.template_id
                """
            )
        )


def create_session(
    session: dict[str, Any], database_path: str | Path = DATABASE_PATH
) -> int:
    now = local_now().isoformat()
    starts_at = session["starts_at"]
    if isinstance(starts_at, datetime):
        starts_at = starts_at.astimezone().isoformat()
    with connection(database_path) as database:
        cursor = database.execute(
            """
            INSERT INTO class_sessions(
                course_code, course_name, venue, lecturer, starts_at,
                grace_minutes, active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                session["course_code"].strip().upper(),
                session["course_name"].strip(),
                session["venue"].strip(),
                session["lecturer"].strip(),
                str(starts_at),
                int(session.get("grace_minutes", 15)),
                now,
            ),
        )
        session_id = int(cursor.lastrowid)
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            (
                "session_created",
                json.dumps({"session_id": session_id, "course_code": session["course_code"]}),
                now,
            ),
        )
    return session_id


def close_session(session_id: int, database_path: str | Path = DATABASE_PATH) -> None:
    now = local_now().isoformat()
    with connection(database_path) as database:
        database.execute(
            "UPDATE class_sessions SET active = 0, ends_at = ? WHERE session_id = ?",
            (now, int(session_id)),
        )
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            ("session_closed", json.dumps({"session_id": int(session_id)}), now),
        )


def list_sessions(
    active_only: bool = False, database_path: str | Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    where = "WHERE cs.active = 1" if active_only else ""
    with connection(database_path) as database:
        return _rows(
            database.execute(
                f"""
                SELECT cs.*,
                       COUNT(a.attendance_id) AS attendance_count,
                       SUM(CASE WHEN a.attendance_status = 'Late' THEN 1 ELSE 0 END) AS late_count
                FROM class_sessions cs
                LEFT JOIN attendance a ON a.session_id = cs.session_id
                {where}
                GROUP BY cs.session_id
                ORDER BY cs.starts_at DESC
                """
            )
        )


def get_session(session_id: int, database_path: str | Path = DATABASE_PATH) -> dict[str, Any] | None:
    with connection(database_path) as database:
        row = database.execute(
            "SELECT * FROM class_sessions WHERE session_id = ?", (int(session_id),)
        ).fetchone()
    return dict(row) if row else None


def mark_attendance(
    session_id: int,
    student_id: str,
    attendance_status: str,
    similarity: float,
    capture_quality: float,
    processing_ms: float,
    database_path: str | Path = DATABASE_PATH,
) -> tuple[bool, dict[str, Any]]:
    """Record attendance once per student/session and return existing data on duplicates."""

    marked_at = local_now().isoformat()
    with connection(database_path) as database:
        existing = database.execute(
            "SELECT * FROM attendance WHERE session_id = ? AND student_id = ?",
            (int(session_id), student_id.strip().upper()),
        ).fetchone()
        if existing:
            return False, dict(existing)

        cursor = database.execute(
            """
            INSERT INTO attendance(
                session_id, student_id, marked_at, attendance_status,
                similarity, capture_quality, processing_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(session_id),
                student_id.strip().upper(),
                marked_at,
                attendance_status,
                float(similarity),
                float(capture_quality),
                float(processing_ms),
            ),
        )
        attendance_id = int(cursor.lastrowid)
        database.execute(
            "INSERT INTO audit_log(event_type, details, created_at) VALUES (?, ?, ?)",
            (
                "attendance_marked",
                json.dumps(
                    {
                        "attendance_id": attendance_id,
                        "session_id": int(session_id),
                        "student_id": student_id.strip().upper(),
                        "similarity": round(float(similarity), 4),
                    }
                ),
                marked_at,
            ),
        )
        row = database.execute(
            "SELECT * FROM attendance WHERE attendance_id = ?", (attendance_id,)
        ).fetchone()
    return True, dict(row)


def attendance_records(
    session_id: int | None = None,
    search: str = "",
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    parameters: list[Any] = []
    if session_id is not None:
        conditions.append("a.session_id = ?")
        parameters.append(int(session_id))
    if search.strip():
        conditions.append("(s.student_id LIKE ? OR s.full_name LIKE ? OR s.programme LIKE ?)")
        term = f"%{search.strip()}%"
        parameters.extend([term, term, term])
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with connection(database_path) as database:
        return _rows(
            database.execute(
                f"""
                SELECT a.attendance_id, a.session_id, cs.course_code, cs.course_name,
                       cs.venue, cs.starts_at, s.student_id, s.full_name, s.programme,
                       s.study_year, s.tutorial_group, a.marked_at,
                       a.attendance_status, a.similarity, a.capture_quality,
                       a.processing_ms
                FROM attendance a
                JOIN students s ON s.student_id = a.student_id
                JOIN class_sessions cs ON cs.session_id = a.session_id
                {where}
                ORDER BY a.marked_at DESC
                """,
                parameters,
            )
        )


def session_roster(
    session_id: int, database_path: str | Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    """Return every active student, including those not marked (Absent)."""

    with connection(database_path) as database:
        return _rows(
            database.execute(
                """
                SELECT s.student_id, s.full_name, s.programme, s.study_year,
                       s.tutorial_group, COALESCE(a.attendance_status, 'Absent') AS attendance_status,
                       a.marked_at, a.similarity, a.capture_quality
                FROM students s
                LEFT JOIN attendance a
                    ON a.student_id = s.student_id AND a.session_id = ?
                WHERE s.status = 'Active'
                ORDER BY s.full_name COLLATE NOCASE
                """,
                (int(session_id),),
            )
        )


def dashboard_statistics(database_path: str | Path = DATABASE_PATH) -> dict[str, Any]:
    today = local_now().date().isoformat()
    with connection(database_path) as database:
        student_count = int(
            database.execute("SELECT COUNT(*) FROM students WHERE status = 'Active'").fetchone()[0]
        )
        active_sessions = int(
            database.execute("SELECT COUNT(*) FROM class_sessions WHERE active = 1").fetchone()[0]
        )
        attendance_today = int(
            database.execute(
                "SELECT COUNT(*) FROM attendance WHERE substr(marked_at, 1, 10) = ?", (today,)
            ).fetchone()[0]
        )
        total_sessions = int(database.execute("SELECT COUNT(*) FROM class_sessions").fetchone()[0])
        average_similarity = database.execute("SELECT AVG(similarity) FROM attendance").fetchone()[0]
    return {
        "students": student_count,
        "active_sessions": active_sessions,
        "attendance_today": attendance_today,
        "total_sessions": total_sessions,
        "average_similarity": float(average_similarity or 0),
    }


def recent_audit_events(
    limit: int = 30, database_path: str | Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    with connection(database_path) as database:
        return _rows(
            database.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (int(limit),)
            )
        )
