"""Where recordings come from.

Today: a local folder (phone recordings copied over). Later: the device's
microSD exposed over USB (mass storage or a serial protocol), which will be
another AudioSource that also knows each file's real start time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus", ".aac"}


@dataclass
class AudioFile:
    path: Path
    started_at: str | None = None  # ISO 8601, if the source knows it


class AudioSource(Protocol):
    def list_files(self) -> Iterator[AudioFile]: ...


class FolderSource:
    def __init__(self, root: str | Path, recursive: bool = True):
        self.root = Path(root)
        self.recursive = recursive

    def list_files(self) -> Iterator[AudioFile]:
        pattern = "**/*" if self.recursive else "*"
        for p in sorted(self.root.glob(pattern)):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield AudioFile(p)
