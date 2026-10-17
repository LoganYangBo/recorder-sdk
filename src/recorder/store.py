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
    """
    ALTER TABLE segments ADD COLUMN speaker_score REAL;
    CREATE TABLE voiceprints (
        name        TEXT PRIMARY KEY,
        model       TEXT NOT NULL,
        embedding   BLOB NOT NULL,  -- float32 vector
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    ALTER TABLE segments ADD COLUMN speaker_match TEXT;
    UPDATE segments SET speaker_match = 'me' WHERE speaker_score IS NOT NULL;
    """,
    """
    ALTER TABLE recordings ADD COLUMN title TEXT;
    """,
    """
    ALTER TABLE words ADD COLUMN original TEXT;
    CREATE TABLE corrections (
        id            INTEGER PRIMARY KEY,
        recording_id  INTEGER REFERENCES recordings(id) ON DELETE SET NULL,
        word_id       INTEGER,  -- may outlive the word (e.g. after --reprocess)
        heard         TEXT NOT NULL,
        chosen        TEXT NOT NULL,
        start         REAL NOT NULL,
        end           REAL NOT NULL,
        speaker       TEXT,
        sentence      TEXT NOT NULL,  -- the segment's text after the correction
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    -- suggested corrections for an unsure word, as a JSON list; NULL = not analysed yet
    ALTER TABLE words ADD COLUMN candidates TEXT;
    """,
    """
    -- word times: 0 = Whisper's, 1 = forced-aligned, -1 = alignment tried, kept Whisper's
    ALTER TABLE words ADD COLUMN aligned INTEGER NOT NULL DEFAULT 0;
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)

_REC_COLS = "r.id, r.path, r.sha256, r.started_at, r.duration, r.language, r.params, r.title"
_SEG_COLS = (
    "s.start, s.end, s.text, s.speaker, s.avg_logprob, s.no_speech_prob, "
    "s.speaker_score, s.speaker_match"
)


def _rec(row) -> Recording:
    return Recording(*row[:6], params=json.loads(row[6]), title=row[7])


def _statements(script: str) -> list[str]:
    stmts, buf = [], ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            stmts.append(buf.strip())
            buf = ""
    return stmts


def migrate(conn: sqlite3.Connection) -> None:
    """Bring the schema up to date. Safe when several connections open the
    same database at once (e.g. the web UI's request threads): each step takes
    the write lock, re-reads the version, and runs only if still needed.
    """
    saved, conn.isolation_level = conn.isolation_level, None  # manage transactions here
    try:
        while True:
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"database schema v{version} is newer than this SDK (v{SCHEMA_VERSION})"
                    )
                if version == SCHEMA_VERSION:
                    conn.execute("COMMIT")
                    return
                for stmt in _statements(MIGRATIONS[version]):
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {version + 1}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
    finally:
        conn.isolation_level = saved


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
                    "(recording_id, start, end, text, speaker, avg_logprob, no_speech_prob, "
                    " speaker_score, speaker_match) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rid,
                        s.start,
                        s.end,
                        s.text,
                        s.speaker,
                        s.avg_logprob,
                        s.no_speech_prob,
                        s.speaker_score,
                        s.speaker_match,
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO words (segment_id, start, end, word, probability) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(cur.lastrowid, w.start, w.end, w.word, w.probability) for w in s.words],
                )
        return rid

    def recording(self, recording_id: int) -> Recording | None:
        row = self.conn.execute(
            f"SELECT {_REC_COLS} FROM recordings r WHERE r.id = ?", (recording_id,)
        ).fetchone()
        return _rec(row) if row else None

    def delete(self, recording_id: int) -> bool:
        """Remove a recording and its segments and words. The audio file is left alone."""
        with self.conn:
            cur = self.conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
        return cur.rowcount > 0

    def rename(self, recording_id: int, title: str | None) -> bool:
        """Set a recording's display name; empty or None goes back to the file name."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE recordings SET title = ? WHERE id = ?",
                ((title or "").strip() or None, recording_id),
            )
        return cur.rowcount > 0

    def recordings(self) -> list[Recording]:
        rows = self.conn.execute(f"SELECT {_REC_COLS} FROM recordings r ORDER BY r.id").fetchall()
        return [_rec(r) for r in rows]

    def segments(self, recording_id: int, words: bool = False) -> list[Segment]:
        rows = self.conn.execute(
            f"SELECT s.id, {_SEG_COLS} FROM segments s WHERE s.recording_id = ? "
            "ORDER BY s.start, s.id",
            (recording_id,),
        ).fetchall()
        segs = {r[0]: Segment(*r[1:]) for r in rows}
        if words:
            for seg_id, *w in self.conn.execute(
                "SELECT w.segment_id, w.start, w.end, w.word, w.probability, w.id, w.original "
                "FROM words w JOIN segments s ON s.id = w.segment_id "
                "WHERE s.recording_id = ? ORDER BY w.start, w.id",
                (recording_id,),
            ):
                segs[seg_id].words.append(Word(*w))
        return list(segs.values())

    def word_context(self, word_id: int) -> dict | None:
        """A word with its segment and recording: {word, recording_id,
        segment_id, segment_start, segment_end, speaker, sentence}."""
        row = self.conn.execute(
            "SELECT w.start, w.end, w.word, w.probability, w.id, w.original, "
            "       s.recording_id, s.id, s.start, s.end, s.speaker, s.text "
            "FROM words w JOIN segments s ON s.id = w.segment_id WHERE w.id = ?",
            (word_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("recording_id", "segment_id", "segment_start", "segment_end", "speaker", "sentence")
        return {"word": Word(*row[:6]), **dict(zip(keys, row[6:]))}

    @staticmethod
    def _pending(align: bool) -> str:
        return "(w.candidates IS NULL OR w.aligned = 0)" if align else "w.candidates IS NULL"

    def unsure_words(
        self, recording_id: int, below: float = 0.6, pending: bool = True, align: bool = False
    ) -> list[tuple[int, float, float, Word]]:
        """Words of a recording Whisper was unsure about (probability below
        `below`) as (segment_id, segment start, segment end, word); by
        default only those without candidates yet (or, with `align`, not
        aligned yet either)."""
        rows = self.conn.execute(
            "SELECT s.id, s.start, s.end, w.start, w.end, w.word, w.probability, w.id, w.original "
            "FROM words w JOIN segments s ON s.id = w.segment_id "
            "WHERE s.recording_id = ? AND w.probability < ? "
            + (f"AND {self._pending(align)} " if pending else "")
            + "ORDER BY w.start, w.id",
            (recording_id, below),
        ).fetchall()
        return [(r[0], r[1], r[2], Word(*r[3:])) for r in rows]

    def pending_unsure(self, below: float = 0.6, align: bool = False) -> dict[int, int]:
        """{recording_id: number of unsure words still to analyse}."""
        return dict(
            self.conn.execute(
                "SELECT s.recording_id, COUNT(*) FROM words w "
                "JOIN segments s ON s.id = w.segment_id "
                f"WHERE w.probability < ? AND {self._pending(align)} GROUP BY s.recording_id",
                (below,),
            ).fetchall()
        )

    def segment_words(self, segment_id: int) -> list[Word]:
        return [
            Word(*r)
            for r in self.conn.execute(
                "SELECT start, end, word, probability, id, original FROM words "
                "WHERE segment_id = ? ORDER BY start, id",
                (segment_id,),
            )
        ]

    def segment_aligned(self, segment_id: int) -> bool:
        """Whether alignment has been tried for every word of a segment."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM words WHERE segment_id = ? AND aligned = 0", (segment_id,)
        ).fetchone()
        return row[0] == 0

    def set_word_times(self, times: dict[int, tuple[float, float] | None]) -> None:
        """Store forced-aligned times ({word_id: (start, end)}); None marks a
        word as tried but kept at Whisper's times."""
        with self.conn:
            for wid, t in times.items():
                if t is None:
                    self.conn.execute("UPDATE words SET aligned = -1 WHERE id = ?", (wid,))
                    continue
                self.conn.execute(
                    "UPDATE words SET start = ?, end = ?, aligned = 1 WHERE id = ?", (*t, wid)
                )
                self.conn.execute(
                    "UPDATE corrections SET start = ?, end = ? WHERE word_id = ?", (*t, wid)
                )

    def set_candidates(self, word_id: int, candidates: list[str]) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE words SET candidates = ? WHERE id = ?", (json.dumps(candidates), word_id)
            )

    def candidates(self, word_id: int) -> list[str] | None:
        row = self.conn.execute("SELECT candidates FROM words WHERE id = ?", (word_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def correct_word(self, word_id: int, chosen: str) -> dict:
        """Replace a word with what the user says was meant (or confirm it, if
        `chosen` is the same word), rebuild its segment's text, and log it in
        `corrections`. Returns the correction as a dict."""
        from .learn import normalize, recase

        ctx = self.word_context(word_id)
        if ctx is None:
            raise KeyError(f"no word {word_id}")
        w = ctx["word"]
        heard = w.original if w.original is not None else w.word
        new = recase(chosen.strip(), w.word) if normalize(chosen) != normalize(w.word) else w.word
        with self.conn:
            self.conn.execute(
                "UPDATE words SET word = ?, original = ? WHERE id = ?", (new, heard, word_id)
            )
            words = [
                r[0]
                for r in self.conn.execute(
                    "SELECT word FROM words WHERE segment_id = ? ORDER BY start, id",
                    (ctx["segment_id"],),
                )
            ]
            sentence = " ".join(words)
            self.conn.execute(
                "UPDATE segments SET text = ? WHERE id = ?", (sentence, ctx["segment_id"])
            )
            # one entry per word: choosing again replaces the earlier choice
            self.conn.execute("DELETE FROM corrections WHERE word_id = ?", (word_id,))
            cur = self.conn.execute(
                "INSERT INTO corrections "
                "(recording_id, word_id, heard, chosen, start, end, speaker, sentence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ctx["recording_id"],
                    word_id,
                    normalize(heard),
                    normalize(chosen),
                    w.start,
                    w.end,
                    ctx["speaker"],
                    sentence,
                ),
            )
        return self.corrections(cur.lastrowid)[0]

    def corrections(self, correction_id: int | None = None) -> list[dict]:
        """The correction log, newest first (or one entry)."""
        where, args = ("WHERE c.id = ?", (correction_id,)) if correction_id else ("", ())
        rows = self.conn.execute(
            "SELECT c.id, c.recording_id, c.word_id, c.heard, c.chosen, c.start, c.end, "
            "       c.speaker, c.sentence, c.created_at, r.title, r.path "
            f"FROM corrections c LEFT JOIN recordings r ON r.id = c.recording_id {where} "
            "ORDER BY c.id DESC",
            args,
        ).fetchall()
        keys = (
            "id",
            "recording_id",
            "word_id",
            "heard",
            "chosen",
            "start",
            "end",
            "speaker",
            "sentence",
            "created_at",
        )
        out = []
        for r in rows:
            d = dict(zip(keys, r[:10]))
            d["recording"] = r[10] or (Path(r[11]).name if r[11] else None)
            out.append(d)
        return out

    def set_speakers(self, recording_id: int, segments: list[Segment]) -> None:
        """Update speaker labels of a recording's segments, given in `segments()` order."""
        ids = [
            r[0]
            for r in self.conn.execute(
                "SELECT id FROM segments WHERE recording_id = ? ORDER BY start, id",
                (recording_id,),
            )
        ]
        if len(ids) != len(segments):
            raise ValueError(
                f"recording {recording_id} has {len(ids)} segments, got {len(segments)}"
            )
        with self.conn:
            self.conn.executemany(
                "UPDATE segments SET speaker = ?, speaker_score = ?, speaker_match = ? "
                "WHERE id = ?",
                [(s.speaker, s.speaker_score, s.speaker_match, i) for s, i in zip(segments, ids)],
            )

    def save_voiceprint(self, name: str, model: str, embedding: bytes) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO voiceprints (name, model, embedding) VALUES (?, ?, ?)",
                (name, model, embedding),
            )

    def voiceprints(self) -> dict[str, tuple[str, bytes]]:
        """All enrolled voices: name -> (model, float32 embedding bytes)."""
        rows = self.conn.execute(
            "SELECT name, model, embedding FROM voiceprints ORDER BY name"
        ).fetchall()
        return {name: (model, emb) for name, model, emb in rows}

    def delete_voiceprint(self, name: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM voiceprints WHERE name = ?", (name,))
        return cur.rowcount > 0

    def voiceprint(self, name: str = "me") -> tuple[str, bytes] | None:
        """(model, float32 embedding bytes) for an enrolled voice, or None."""
        row = self.conn.execute(
            "SELECT model, embedding FROM voiceprints WHERE name = ?", (name,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def search(self, query: str, limit: int = 50) -> list[tuple[Recording, Segment]]:
        rows = self.conn.execute(
            f"SELECT {_REC_COLS}, {_SEG_COLS} "
            "FROM segments s JOIN recordings r ON r.id = s.recording_id "
            "WHERE s.text LIKE ? ORDER BY r.id, s.start LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [(_rec(r[:8]), Segment(*r[8:])) for r in rows]
