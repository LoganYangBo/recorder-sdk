"""The interface for apps built on the SDK (English practice, daily review,
...). It answers questions in terms of people, days and words, so apps never
touch the database tables, which can change between versions.

    from recorder import Recorder

    with Recorder("recorder.db") as rec:
        for u in rec.my_utterances(day="2026-10-16"):
            print(u.at, u.text)
        for w in rec.unsure_words(speaker="me", since="2026-10-01"):
            print(w.word, w.candidates, w.sentence)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .store import Store
from .turns import merge_turns

When = str | date | datetime | None


def _iso(t: When) -> str | None:
    if t is None:
        return None
    if isinstance(t, datetime):
        return t.replace(microsecond=0, tzinfo=None).isoformat()
    if isinstance(t, date):
        return t.isoformat() + "T00:00:00"
    return t if "T" in t else t + "T00:00:00"


def _day_range(day: When) -> tuple[str, str]:
    d = date.fromisoformat(day[:10]) if isinstance(day, str) else day
    if isinstance(d, datetime):
        d = d.date()
    return d.isoformat() + "T00:00:00", (d + timedelta(days=1)).isoformat() + "T00:00:00"


def _at(started_at: str | None, offset: float) -> datetime | None:
    if not started_at:
        return None
    return datetime.fromisoformat(started_at) + timedelta(seconds=offset)


@dataclass
class Utterance:
    """What one person said without interruption (consecutive segments merged)."""

    recording_id: int
    speaker: str | None  # "me", an enrolled name, "other", or None if not labelled
    start: float  # seconds into the recording
    end: float
    text: str
    at: datetime | None  # wall-clock start, if the recording's start time is known

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class UnsureWord:
    """A word Whisper wasn't sure about, with what it might have been."""

    word_id: int
    word: str  # as transcribed (or as corrected)
    probability: float
    candidates: list[str] | None  # None until analysed (see ops.suggest_words)
    corrected_from: str | None  # what Whisper heard, if the user corrected or confirmed it
    sentence: str
    speaker: str | None
    recording_id: int
    start: float
    end: float
    at: datetime | None
    # whose mistake a correction was: "misheard" (Whisper's) / "mispronounced" (the
    # speaker's) / None (confirmed, not corrected, or the user didn't say)
    reason: str | None = None


@dataclass
class Correction:
    word: str  # what was meant
    heard: str  # what Whisper heard (the same, if the user confirmed it)
    sentence: str
    speaker: str | None
    recording_id: int | None  # None once the recording is deleted
    start: float
    end: float
    created_at: str
    reason: str | None = None  # "misheard" / "mispronounced" / None (see UnsureWord.reason)

    @property
    def confirmed(self) -> bool:
        return self.word == self.heard


@dataclass
class Conversation:
    """Talk that runs on without a long break, across the recorder's 10-minute files."""

    start: datetime
    end: datetime
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def recordings(self) -> list[int]:
        return sorted({u.recording_id for u in self.utterances})

    @property
    def speakers(self) -> dict[str, float]:
        """Seconds spoken by each speaker ("unlabelled" for None), most first."""
        out: dict[str, float] = {}
        for u in self.utterances:
            key = u.speaker or "unlabelled"
            out[key] = out.get(key, 0.0) + u.duration
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @property
    def text(self) -> str:
        return "\n".join(f"{u.speaker or '?'}: {u.text}" for u in self.utterances)


@dataclass
class Changes:
    changed: list[int]  # recordings added, re-transcribed, corrected, relabelled, renamed...
    deleted: list[int]
    cursor: int  # pass to changes() next time


@dataclass
class Day:
    date: date
    recordings: int
    seconds: float  # recorded audio
    speakers: dict[str, int] = field(default_factory=dict)  # segments per speaker


class Recorder:
    """Read access to everything the SDK has stored, for apps."""

    def __init__(self, db: str | Path = "recorder.db"):
        self.store = Store(db)

    def close(self) -> None:
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- recordings and days ----

    def recordings(self, since: When = None, until: When = None, day: When = None):
        """Recordings (store.Recording) that started in [since, until), or on `day`."""
        if day is not None:
            since, until = _day_range(day)
        lo, hi = _iso(since), _iso(until)
        if lo is None and hi is None:
            return self.store.recordings()
        return [
            r
            for r in self.store.recordings()
            if r.started_at
            and (lo is None or r.started_at >= lo)
            and (hi is None or r.started_at < hi)
        ]

    def days(self) -> list[Day]:
        """Each day that has recordings (by start time), oldest first."""
        out: dict[str, Day] = {}
        rows = self.store.conn.execute(
            "SELECT substr(r.started_at, 1, 10), r.id, r.duration, s.speaker, COUNT(s.id) "
            "FROM recordings r LEFT JOIN segments s ON s.recording_id = r.id "
            "WHERE r.started_at IS NOT NULL GROUP BY r.id, s.speaker"
        ).fetchall()
        seen: set[int] = set()
        for d, rid, duration, speaker, n in rows:
            day = out.setdefault(d, Day(date.fromisoformat(d), 0, 0.0))
            if rid not in seen:
                seen.add(rid)
                day.recordings += 1
                day.seconds += duration
            if n:
                key = speaker or "unlabelled"
                day.speakers[key] = day.speakers.get(key, 0) + n
        return [out[k] for k in sorted(out)]

    # ---- what was said ----

    def utterances(
        self,
        speaker: str | None = None,
        day: When = None,
        since: When = None,
        until: When = None,
        recording: int | None = None,
    ) -> list[Utterance]:
        """Speaker turns, in time order, optionally one speaker's, on one day,
        in [since, until) (by wall-clock time) or from one recording."""
        if day is not None:
            since, until = _day_range(day)
        lo, hi = _iso(since), _iso(until)
        if recording is not None:
            rec = self.store.recording(recording)
            recs = [rec] if rec else []
        elif lo or hi:
            recs = [r for r in self.store.recordings() if r.started_at]
        else:
            recs = self.store.recordings()
        out = []
        for r in recs:
            if r.started_at and (lo or hi):  # skip recordings entirely outside the range
                end = _at(r.started_at, r.duration).isoformat()
                if (hi and r.started_at >= hi) or (lo and end < lo):
                    continue
            for t in merge_turns(self.store.segments(r.id)):
                at = _at(r.started_at, t.start)
                if speaker is not None and t.speaker != speaker:
                    continue
                if at is not None and (
                    (lo and at.isoformat() < lo) or (hi and at.isoformat() >= hi)
                ):
                    continue
                out.append(Utterance(r.id, t.speaker, t.start, t.end, t.text, at))
        out.sort(key=lambda u: (u.at or datetime.max, u.recording_id, u.start))  # undated last
        return out

    def my_utterances(self, **filters) -> list[Utterance]:
        """What the wearer said; same filters as utterances()."""
        return self.utterances(speaker="me", **filters)

    def search(
        self,
        query: str,
        speaker: str | None = None,
        since: When = None,
        until: When = None,
        day: When = None,
        limit: int = 50,
    ) -> list[Utterance]:
        """Segments containing every word of `query` (any language)."""
        if day is not None:
            since, until = _day_range(day)
        hits = self.store.search(query, limit, speaker, _iso(since), _iso(until))
        return [
            Utterance(r.id, s.speaker, s.start, s.end, s.text, _at(r.started_at, s.start))
            for r, s in hits
        ]

    # ---- words ----

    def unsure_words(
        self,
        speaker: str | None = None,
        since: When = None,
        until: When = None,
        day: When = None,
        recording: int | None = None,
        below: float = 0.6,
        include_reviewed: bool = False,
    ) -> list[UnsureWord]:
        """Words Whisper was less than `below` sure about, with candidate
        corrections; ones the user already corrected or confirmed are left
        out unless `include_reviewed`."""
        if day is not None:
            since, until = _day_range(day)
        where, args = ["w.probability < ?"], [below]
        if not include_reviewed:
            where.append("w.original IS NULL")
        if speaker is not None:
            where.append("s.speaker = ?")
            args.append(speaker)
        if recording is not None:
            where.append("r.id = ?")
            args.append(recording)
        if since is not None:
            where.append("r.started_at >= ?")
            args.append(_iso(since))
        if until is not None:
            where.append("r.started_at < ?")
            args.append(_iso(until))
        rows = self.store.conn.execute(
            "SELECT w.id, w.word, w.probability, w.candidates, w.original, s.text, s.speaker, "
            "       r.id, w.start, w.end, r.started_at, c.reason "
            "FROM words w JOIN segments s ON s.id = w.segment_id "
            "JOIN recordings r ON r.id = s.recording_id "
            "LEFT JOIN corrections c ON c.word_id = w.id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY r.started_at IS NULL, r.started_at, r.id, w.start",
            args,
        ).fetchall()
        return [
            UnsureWord(
                wid,
                word,
                p,
                json.loads(c) if c is not None else None,
                orig,
                sentence,
                spk,
                rid,
                start,
                end,
                _at(started, start),
                reason,
            )  # fmt: skip
            for wid, word, p, c, orig, sentence, spk, rid, start, end, started, reason in rows
        ]

    def corrections(self, include_confirmed: bool = True) -> list[Correction]:
        """The user's corrections (newest first); confirmed words too unless told not to."""
        out = []
        for c in self.store.corrections():
            corr = Correction(
                c["chosen"], c["heard"], c["sentence"], c["speaker"], c["recording_id"],
                c["start"], c["end"], c["created_at"], c["reason"],
            )  # fmt: skip
            if include_confirmed or not corr.confirmed:
                out.append(corr)
        return out

    # ---- conversations and changes ----

    def conversations(
        self,
        day: When = None,
        since: When = None,
        until: When = None,
        gap: float = 300.0,
        min_seconds: float = 0.0,
    ) -> list[Conversation]:
        """What was said, grouped into conversations: utterances (from any number of
        recordings) with no silence longer than `gap` seconds between them. Only
        recordings with a known start time count. Conversations with less than
        `min_seconds` of speech are left out."""
        utts = [u for u in self.utterances(day=day, since=since, until=until) if u.at]
        out: list[Conversation] = []
        last_end: datetime | None = None
        for u in utts:
            end = u.at + timedelta(seconds=u.duration)
            if last_end is None or (u.at - last_end).total_seconds() > gap:
                out.append(Conversation(u.at, end))
            conv = out[-1]
            conv.utterances.append(u)
            conv.end = max(conv.end, end)
            last_end = conv.end
        return [c for c in out if sum(u.duration for u in c.utterances) >= min_seconds]

    def changes(self, since: int = 0) -> Changes:
        """Recordings added or changed (transcribed again, corrected, relabelled,
        renamed, re-dated) and deleted since cursor `since` (0: everything). Keep the
        returned cursor and pass it next time, instead of re-reading everything."""
        changed, deleted, cursor = self.store.changes(since)
        return Changes(changed, deleted, cursor)

    # ---- vocab ----

    def vocab(self) -> list[str]:
        """The user's names and jargon (newest first), as given to Whisper."""
        return [v["term"] for v in self.store.vocab()]
