from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float  # Whisper's confidence for this word, 0..1
    id: int | None = None  # database id, when loaded from a Store
    original: str | None = None  # what Whisper heard, if the user corrected or confirmed it


@dataclass
class Segment:
    start: float  # seconds from the start of the recording
    end: float
    text: str
    speaker: str | None = None  # "me" / an enrolled name / "other" / None (not labelled)
    avg_logprob: float | None = None  # segment confidence; below about -1 is usually unreliable
    no_speech_prob: float | None = None
    speaker_score: float | None = None  # similarity to the best-matching voiceprint, -1..1
    speaker_match: str | None = None  # that voiceprint's name, even if below the threshold
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    params: dict = field(default_factory=dict)  # transcriber settings, stored for re-runs

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
    params: dict = field(default_factory=dict)  # how it was transcribed (model, prompt, ...)
    title: str | None = None  # display name set by the user; the file itself is not renamed

    @property
    def name(self) -> str:
        return self.title or Path(self.path).name
