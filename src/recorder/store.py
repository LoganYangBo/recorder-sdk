from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Recording, Segment, Transcript

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    started_at  TEXT,
    duration    REAL NOT NULL,
    language    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY,
    recording_id  INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    start         REAL NOT NULL,
    end           REAL NOT NULL,
    text          TEXT NOT NULL,
    speaker       TEXT
);
CREATE INDEX IF NOT EXISTS idx_segments_recording ON segments(recording_id);
"""


class Store:
    def __init__(self, db_path: str | Path = "recorder.db"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def find_by_hash(self, sha256: str) -> Recording | None:
        row = self.conn.execute(
            "SELECT id, path, sha256, started_at, duration, language "
            "FROM recordings WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        return Recording(*row) if row else None

    def save(
        self, path: str | Path, sha256: str, transcript: Transcript, started_at: str | None = None
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO recordings (path, sha256, started_at, duration, language) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(path), sha256, started_at, transcript.duration, transcript.language),
            )
            rid = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO segments (recording_id, start, end, text, speaker) "
                "VALUES (?, ?, ?, ?, ?)",
                [(rid, s.start, s.end, s.text, s.speaker) for s in transcript.segments],
            )
        return rid

    def recordings(self) -> list[Recording]:
        rows = self.conn.execute(
            "SELECT id, path, sha256, started_at, duration, language FROM recordings ORDER BY id"
        ).fetchall()
        return [Recording(*r) for r in rows]

    def segments(self, recording_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT start, end, text, speaker FROM segments WHERE recording_id = ? ORDER BY start",
            (recording_id,),
        ).fetchall()
        return [Segment(*r) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[tuple[Recording, Segment]]:
        rows = self.conn.execute(
            "SELECT r.id, r.path, r.sha256, r.started_at, r.duration, r.language, "
            "       s.start, s.end, s.text, s.speaker "
            "FROM segments s JOIN recordings r ON r.id = s.recording_id "
            "WHERE s.text LIKE ? ORDER BY r.id, s.start LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [(Recording(*r[:6]), Segment(*r[6:])) for r in rows]
