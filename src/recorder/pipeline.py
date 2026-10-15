from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .speaker import NoopLabeler, SpeakerLabeler
from .store import Store
from .sync import AudioSource
from .transcribe import Transcriber


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Result:
    recording_id: int
    path: Path
    skipped: bool  # True if this file was already in the database


class Pipeline:
    """audio file -> transcribe -> label speakers -> store.

    Files are identified by content hash, so re-running on the same
    folder (or a re-synced device) does not transcribe anything twice.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        store: Store,
        labeler: SpeakerLabeler | None = None,
    ):
        self.transcriber = transcriber
        self.store = store
        self.labeler = labeler or NoopLabeler()

    def process_file(self, path: str | Path, started_at: str | None = None) -> Result:
        path = Path(path)
        digest = file_sha256(path)
        existing = self.store.find_by_hash(digest)
        if existing:
            return Result(existing.id, path, skipped=True)
        transcript = self.transcriber.transcribe(path)
        transcript = self.labeler.label(path, transcript)
        rid = self.store.save(path, digest, transcript, started_at=started_at)
        return Result(rid, path, skipped=False)

    def process_source(self, source: AudioSource) -> list[Result]:
        return [self.process_file(f.path, f.started_at) for f in source.list_files()]
