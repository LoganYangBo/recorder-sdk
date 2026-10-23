from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .models import Transcript
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
    recording_id: int | None
    path: Path
    skipped: bool  # True if this file was already in the database
    error: str | None = None  # set (and recording_id None) if processing failed
    duration: float = 0.0  # seconds of audio
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage

    def timing(self) -> str:
        """'600 s of audio: decode 1.2 s, transcribe 21.0 s (29x), speakers 2.1 s'"""
        parts = []
        for stage, t in self.timings.items():
            speed = f" ({self.duration / t:.0f}x)" if stage == "transcribe" and t > 0 else ""
            parts.append(f"{stage} {t:.1f} s{speed}")
        return f"{self.duration:.0f} s of audio: " + ", ".join(parts)


class Pipeline:
    """audio file -> transcribe -> label speakers -> store.

    Files are identified by content hash, so re-running on the same
    folder (or a re-synced device) does not transcribe anything twice.
    With reprocess=True, files already in the database are transcribed
    again and overwritten in place (same recording id), e.g. after
    switching models or adding hotwords.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        store: Store,
        labeler: SpeakerLabeler | None = None,
        reprocess: bool = False,
    ):
        self.transcriber = transcriber
        self.store = store
        self.labeler = labeler or NoopLabeler()
        self.reprocess = reprocess
        self._done: set[str] = set()  # hashes handled in this run, so duplicates aren't redone

    def process_file(
        self,
        path: str | Path,
        started_at: str | None = None,
        digest: str | None = None,
        stored_path: str | Path | None = None,
    ) -> Result:
        """Transcribe, label and store one file. `digest` identifies the
        content (default: the file's sha256; sync passes the hash of the
        original when it stores a compressed copy). `stored_path` is where the
        recording will live, if not at `path` (sync transcribes the WAV while
        its compressed copy is still being made).

        The audio is decoded once and shared by Whisper and the speaker
        labeller, when both can take samples."""
        path = Path(path)
        digest = digest or file_sha256(path)
        existing = self.store.find_by_hash(digest)
        if existing and (not self.reprocess or digest in self._done):
            return Result(existing.id, path, skipped=True)
        timings: dict[str, float] = {}
        t = time.perf_counter()
        load = getattr(self.transcriber, "load_audio", None)
        label_audio = getattr(self.labeler, "label_audio", None)
        audio = load(path) if load else None
        if audio is not None:
            timings["decode"] = time.perf_counter() - t
            t = time.perf_counter()
        transcript = self.transcriber.transcribe(path if audio is None else audio)
        timings["transcribe"] = time.perf_counter() - t
        if not isinstance(self.labeler, NoopLabeler):
            t = time.perf_counter()
            if audio is not None and label_audio:
                transcript = label_audio(audio, transcript)
            else:
                transcript = self.labeler.label(path, transcript)
            timings["speakers"] = time.perf_counter() - t
        rid = self.store.save(
            stored_path or path,
            digest,
            transcript,
            started_at=started_at,
            replace_id=existing.id if existing else None,
        )
        self._done.add(digest)
        return Result(rid, path, False, duration=transcript.duration, timings=timings)

    def process_source(self, source: AudioSource) -> list[Result]:
        """Process every file; one failing (corrupt, unreadable) doesn't stop
        the rest: its Result carries the error instead."""
        results = []
        for f in source.list_files():
            try:
                results.append(self.process_file(f.path, f.started_at))
            except Exception as e:
                results.append(Result(None, f.path, False, f"{type(e).__name__}: {e}"))
        return results

    def relabel(self, recording_id: int) -> None:
        """Re-run speaker labelling on a stored recording without re-transcribing,
        e.g. after enrolling a new voiceprint. Needs the audio file to still exist.
        """
        rec = self.store.recording(recording_id)
        if rec is None:
            raise KeyError(f"no recording {recording_id}")
        segments = self.store.segments(recording_id)
        self.labeler.label(rec.path, Transcript(rec.language, rec.duration, segments))
        self.store.set_speakers(recording_id, segments)
