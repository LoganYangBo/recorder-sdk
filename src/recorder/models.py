from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float  # Whisper's confidence for this word, 0..1


@dataclass
class Segment:
    start: float  # seconds from the start of the recording
    end: float
    text: str
    speaker: str | None = None  # "me" / "other" / None (unknown)
    avg_logprob: float | None = None  # segment confidence; below about -1 is usually unreliable
    no_speech_prob: float | None = None
    speaker_score: float | None = None  # similarity to the wearer's voiceprint, -1..1
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
