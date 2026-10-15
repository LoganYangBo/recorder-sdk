"""Speaker labelling ("me" vs "other"). Interface only for now.

A real implementation will embed each segment's audio (e.g. with
speechbrain / pyannote / resemblyzer), compare it to an enrolled
voiceprint of the wearer, and set Segment.speaker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Transcript


class SpeakerLabeler(Protocol):
    def label(self, path: str | Path, transcript: Transcript) -> Transcript: ...


class NoopLabeler:
    """Leaves every segment's speaker as None."""

    def label(self, path: str | Path, transcript: Transcript) -> Transcript:
        return transcript
