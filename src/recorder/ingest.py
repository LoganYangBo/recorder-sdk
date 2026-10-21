"""Sync: bring recordings from a source (the device, a folder) into the
library and the database, one file at a time.

For each file: hash it; if it is already in the database it is done.
Otherwise copy it into the library as `YYYY/MM/DD/<name>` (WAV is compressed
to Ogg Opus, about a tenth of the size), transcribe and label it, and record
the outcome in the `files` table. A file that fails is logged with its error
and retried on the next sync (up to `max_attempts`), without stopping the
others. Files on the source are deleted only when asked to, and only once
their copy is safely in the library and in the database.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Pipeline, file_sha256
from .store import Store
from .sync import AudioFile, AudioSource

OPUS_BITRATE = 32000


def has_speech(segments) -> bool:
    """Whether a transcript holds anything Whisper actually heard as speech: some text that it
    neither thinks is silence nor made up (noise often comes back empty, or as a stray
    "Thank you." with low confidence)."""
    return any(
        s.text.strip()
        and (s.no_speech_prob is None or s.no_speech_prob < 0.6)
        and (s.avg_logprob is None or s.avg_logprob > -1.0)
        for s in segments
    )  # plenty for 16 kHz speech; Whisper accuracy is unaffected


@dataclass
class SyncResult:
    path: Path  # on the source
    status: str  # new / known / noise / failed / gave-up
    recording_id: int | None = None
    error: str | None = None


def compress_to_opus(src: Path, dest: Path, bitrate: int = OPUS_BITRATE) -> None:
    """Re-encode audio as mono 16 kHz Ogg Opus (needs PyAV)."""
    import av

    with av.open(str(src)) as inp, av.open(str(dest), "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=16000)
        stream.bit_rate = bitrate
        stream.layout = "mono"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in inp.decode(audio=0):
            for f in resampler.resample(frame):
                out.mux(stream.encode(f))
        for f in resampler.resample(None):
            out.mux(stream.encode(f))
        out.mux(stream.encode(None))


def library_path(library: Path, f: AudioFile, compress: bool) -> Path:
    folder = library / (f.started_at[:10].replace("-", "/") if f.started_at else "undated")
    suffix = ".opus" if compress else f.path.suffix.lower()
    path = folder / (f.path.stem + suffix)
    n = 1
    while path.exists():  # same name, different content
        path = folder / f"{f.path.stem}-{n}{suffix}"
        n += 1
    return path


class Syncer:
    def __init__(
        self,
        store: Store,
        pipeline: Pipeline,
        library: str | Path,
        compress: bool = True,
        delete_source: bool = False,
        drop_noise: bool = False,
        max_attempts: int = 3,
        after: Callable[[int], None] | None = None,
        log: Callable[[str], None] = lambda msg: None,
    ):
        self.store = store
        self.pipeline = pipeline
        self.library = Path(library)
        self.compress = compress
        self.delete_source = delete_source
        self.drop_noise = drop_noise  # forget files with no speech (the recorder's false starts)
        self.max_attempts = max_attempts
        self.after = after  # called with each new recording id (e.g. suggest unsure words)
        self.log = log

    def sync(self, source: AudioSource, retry: bool = False) -> list[SyncResult]:
        """Sync every file of `source`; `retry` also retries files that gave up."""
        results = []
        for f in source.list_files():
            r = self.sync_file(f, retry)
            results.append(r)
            if r.status != "known":
                self.log(f"{f.path.name}: {r.status}" + (f" ({r.error})" if r.error else ""))
        return results

    def sync_file(self, f: AudioFile, retry: bool = False) -> SyncResult:
        source = str(f.path)
        try:
            digest = file_sha256(f.path)
        except OSError as e:  # unreadable on the source: nothing to record against
            return SyncResult(f.path, "failed", error=f"{type(e).__name__}: {e}")
        known = self.store.find_by_hash(digest)
        if known:
            self.store.set_file_status(digest, source, "done", recording_id=known.id)
            self._cleanup(f.path, known.path)
            return SyncResult(f.path, "known", known.id)
        status = self.store.file_status(digest)
        if status and status["status"] == "noise":
            self._drop(f.path)
            return SyncResult(f.path, "noise")
        if status and status["status"] == "failed" and not retry:
            if status["attempts"] >= self.max_attempts:
                return SyncResult(f.path, "gave-up", error=status["error"])
        copy = Path(status["path"]) if status and status["path"] else None
        try:
            if copy is None or not copy.is_file():
                copy = self._copy(f)
            self.store.set_file_status(
                digest, source, "pending", path=str(copy), started_at=f.started_at
            )
            r = self.pipeline.process_file(copy, f.started_at, digest=digest)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.store.set_file_status(digest, source, "failed", error=error)
            return SyncResult(f.path, "failed", error=error)
        if self.drop_noise and not has_speech(self.store.segments(r.recording_id)):
            self.store.delete(r.recording_id)
            copy.unlink(missing_ok=True)
            self.store.set_file_status(digest, source, "noise", path=None)
            self._drop(f.path)
            return SyncResult(f.path, "noise")
        self.store.set_file_status(digest, source, "done", recording_id=r.recording_id)
        if self.after:
            try:
                self.after(r.recording_id)
            except Exception as e:  # extras (suggestions) can be redone later
                self.log(f"{f.path.name}: after-sync step failed ({type(e).__name__}: {e})")
        self._cleanup(f.path, str(copy))
        return SyncResult(f.path, "new", r.recording_id)

    def _copy(self, f: AudioFile) -> Path:
        compress = self.compress and f.path.suffix.lower() == ".wav"
        dest = library_path(self.library, f, compress)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            if compress:
                compress_to_opus(f.path, tmp)
            else:
                shutil.copyfile(f.path, tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    def _cleanup(self, source: Path, copy: str) -> None:
        """Delete the file from the source once its library copy exists."""
        if not self.delete_source or Path(copy).resolve() == source.resolve():
            return
        if Path(copy).is_file():
            source.unlink(missing_ok=True)

    def _drop(self, source: Path) -> None:
        """Delete a noise-only file from the source: there is nothing to keep."""
        if self.delete_source:
            source.unlink(missing_ok=True)
