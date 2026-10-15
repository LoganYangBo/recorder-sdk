from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float  # seconds from the start of the recording
    end: float
    text: str
    speaker: str | None = None  # "me" / "other" / None (unknown)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass
class Recording:
    id: int
    path: str
    sha256: str
    started_at: str | None  # ISO 8601; filled by device sync later
    duration: float
    language: str
