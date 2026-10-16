from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Recording, Segment, Transcript, Word

# Each entry upgrades the schema by one version; PRAGMA user_version records
# how many have run. Append new migrations, never edit old ones.
# v1 uses IF NOT EXISTS because databases created before versioning already
# have these tables at user_version 0.
MIGRATIONS = [
    """
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
    """,
    """
    ALTER TABLE recordings ADD COLUMN params TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE segments ADD COLUMN avg_logprob REAL;
    ALTER TABLE segments ADD COLUMN no_speech_prob REAL;
    CREATE TABLE words (
        id          INTEGER PRIMARY KEY,
        segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
        start       REAL NOT NULL,
        end         REAL NOT NULL,
        word        TEXT NOT NULL,
        probability REAL NOT NULL
    );
    CREATE INDEX idx_words_segment ON words(segment_id);
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)

_REC_COLS = "r.id, r.path, r.sha256, r.started_at, r.duration, r.language, r.params"
_SEG_COLS = "s.start, s.end, s.text, s.speaker, s.avg_logprob, s.no_speech_prob"


def _rec(row) -> Recording:
    return Recording(*row[:6], params=json.loads(row[6]))


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema v{version} is newer than this SDK (v{SCHEMA_VERSION})"
        )
    for i in range(version, SCHEMA_VERSION):
        # executescript commits on its own, so each migration and its version
        # bump run as one script inside an explicit transaction
        conn.executescript(f"BEGIN; {MIGRATIONS[i]} PRAGMA user_version = {i + 1}; COMMIT;")


class Store:
    def __init__(self, db_path: str | Path = "recorder.db"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def find_by_hash(self, sha256: str) -> Recording | None:
        row = self.conn.execute(
            f"SELECT {_REC_COLS} FROM recordings r WHERE r.sha256 = ?", (sha256,)
        ).fetchone()
        return _rec(row) if row else None

    def save(
        self,
        path: str | Path,
        sha256: str,
        transcript: Transcript,
        started_at: str | None = None,
        replace_id: int | None = None,
    ) -> int:
        """Insert a recording, or overwrite recording `replace_id` (keeping its id)."""
        params = json.dumps(transcript.params, sort_keys=True)
        with self.conn:
            if replace_id is None:
                cur = self.conn.execute(
                    "INSERT INTO recordings (path, sha256, started_at, duration, language, params) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(path),
                        sha256,
                        started_at,
                        transcript.duration,
                        transcript.language,
                        params,
                    ),
                )
                rid = cur.lastrowid
            else:
                rid = replace_id
                self.conn.execute(
                    "UPDATE recordings SET path = ?, started_at = COALESCE(?, started_at), "
                    "duration = ?, language = ?, params = ? WHERE id = ?",
                    (str(path), started_at, transcript.duration, transcript.language, params, rid),
                )
                self.conn.execute("DELETE FROM segments WHERE recording_id = ?", (rid,))
            for s in transcript.segments:
                cur = self.conn.execute(
                    "INSERT INTO segments "
                    "(recording_id, start, end, text, speaker, avg_logprob, no_speech_prob) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rid, s.start, s.end, s.text, s.speaker, s.avg_logprob, s.no_speech_prob),
                )
                self.conn.executemany(
                    "INSERT INTO words (segment_id, start, end, word, probability) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(cur.lastrowid, w.start, w.end, w.word, w.probability) for w in s.words],
                )
        return rid

    def recordings(self) -> list[Recording]:
        rows = self.conn.execute(f"SELECT {_REC_COLS} FROM recordings r ORDER BY r.id").fetchall()
        return [_rec(r) for r in rows]

    def segments(self, recording_id: int, words: bool = False) -> list[Segment]:
        rows = self.conn.execute(
            f"SELECT s.id, {_SEG_COLS} FROM segments s WHERE s.recording_id = ? ORDER BY s.start",
            (recording_id,),
        ).fetchall()
        segs = {r[0]: Segment(*r[1:]) for r in rows}
        if words:
            for seg_id, *w in self.conn.execute(
                "SELECT w.segment_id, w.start, w.end, w.word, w.probability "
                "FROM words w JOIN segments s ON s.id = w.segment_id "
                "WHERE s.recording_id = ? ORDER BY w.start",
                (recording_id,),
            ):
                segs[seg_id].words.append(Word(*w))
        return list(segs.values())

    def search(self, query: str, limit: int = 50) -> list[tuple[Recording, Segment]]:
        rows = self.conn.execute(
            f"SELECT {_REC_COLS}, {_SEG_COLS} "
            "FROM segments s JOIN recordings r ON r.id = s.recording_id "
            "WHERE s.text LIKE ? ORDER BY r.id, s.start LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [(_rec(r[:7]), Segment(*r[7:])) for r in rows]
