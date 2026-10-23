"""A recording's transcript as a file to keep or share: Markdown (speaker turns
with times), SubRip subtitles (.srt, one cue per segment) or plain text."""

from __future__ import annotations

from datetime import datetime, timedelta

from .models import Recording, Segment
from .turns import merge_turns

FORMATS = ("md", "srt", "txt")


def _clock(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _when(rec: Recording, offset: float) -> str:
    """Wall-clock time of a point in the recording if its start is known, else the offset."""
    if rec.started_at:
        at = datetime.fromisoformat(rec.started_at) + timedelta(seconds=offset)
        return at.strftime("%H:%M:%S")
    return _clock(offset)


def to_markdown(rec: Recording, segments: list[Segment]) -> str:
    when = (rec.started_at or "").replace("T", " ")[:16]
    lines = [f"# {rec.name}", ""]
    meta = [m for m in (when, f"{_clock(rec.duration)} long", rec.language) if m]
    lines += [" · ".join(meta), ""]
    for t in merge_turns(segments):
        who = f"**{t.speaker}**" if t.speaker else "**?**"
        lines += [f"{who} `{_when(rec, t.start)}`  ", t.text.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def to_srt(segments: list[Segment]) -> str:
    cues = []
    for i, s in enumerate(segments, 1):
        who = f"[{s.speaker}] " if s.speaker else ""
        cues.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{who}{s.text.strip()}\n")
    return "\n".join(cues)


def to_text(segments: list[Segment]) -> str:
    out = []
    for t in merge_turns(segments):
        who = f"{t.speaker}: " if t.speaker else ""
        out.append(f"{who}{t.text.strip()}")
    return "\n".join(out) + "\n"


def export(rec: Recording, segments: list[Segment], fmt: str = "md") -> str:
    if fmt == "md":
        return to_markdown(rec, segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "txt":
        return to_text(segments)
    raise ValueError(f"unknown format {fmt!r} (one of {', '.join(FORMATS)})")
