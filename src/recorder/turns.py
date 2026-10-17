"""Merge Whisper's segments into speaker turns.

VAD and Whisper often split one utterance into several short segments.
Once segments carry speaker labels, consecutive segments by the same
speaker with only a short pause between them are one turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Segment


@dataclass
class Turn:
    start: float
    end: float
    speaker: str | None
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def merge_turns(segments: list[Segment], max_gap: float = 1.5) -> list[Turn]:
    """Group consecutive segments with the same speaker and a pause of at
    most `max_gap` seconds. Segments must be in time order.
    """
    turns: list[Turn] = []
    for s in segments:
        last = turns[-1] if turns else None
        if last and last.speaker == s.speaker and s.start - last.end <= max_gap:
            last.end = max(last.end, s.end)
            last.segments.append(s)
        else:
            turns.append(Turn(s.start, s.end, s.speaker, [s]))
    return turns
