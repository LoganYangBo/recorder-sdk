"""Sync: bring recordings from a source (the device, a folder) into the
library and the database.

For each file: if it was synced before (same place, size and modification
time) it is done without reading it again. Otherwise it is read once, copied
and hashed in the same pass; if the content is already in the database it is
done. Otherwise it is transcribed and labelled from the local copy, and kept
in the library as `YYYY/MM/DD/<name>` (WAV is compressed to Ogg Opus, about a
tenth of the size). The outcome goes in the `files` table. A file that fails
is logged with its error and retried on the next sync (up to `max_attempts`),
without stopping the others. Files on the source are deleted only when asked
to, and only once their copy is safely in the library and in the database.

The stages overlap: while Whisper works on one file (GPU), the next is read
from the source (disk) and the one before is compressed (CPU). Follow-up work
(`after`, e.g. suggestions for unsure words) runs once every file is in, so
transcripts arrive as early as possible.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import Pipeline
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


def library_path(
    library: Path, f: AudioFile, compress: bool, taken: set[Path] | None = None
) -> Path:
    """Where a file goes in the library; `taken` are paths already promised to
    files still on their way."""
    folder = library / (f.started_at[:10].replace("-", "/") if f.started_at else "undated")
    suffix = ".opus" if compress else f.path.suffix.lower()
    path = folder / (f.path.stem + suffix)
    n = 1
    while path.exists() or (taken and path in taken):  # same name, different content
        path = folder / f"{f.path.stem}-{n}{suffix}"
        n += 1
    return path


def copy_and_hash(src: Path, dest: Path) -> str:
    """Copy a file and return its sha256, reading it once."""
    h = hashlib.sha256()
    with open(src, "rb") as inp, open(dest, "wb") as out:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
    return h.hexdigest()


@dataclass
class _Job:
    """One file on its way through a sync."""

    f: AudioFile
    size: int | None = None
    mtime: float | None = None
    digest: str | None = None
    copy: Path | None = None  # the local file to transcribe
    dest: Path | None = None  # where it ends up, when compressed later
    fresh: bool = False  # copy made by this sync (remove it if not needed)
    result: SyncResult | None = None  # decided without transcribing
    error: str | None = None
    fetch: Future | None = None
    stat_known: bool = False  # found by size and mtime: not read at all
    extra: dict = field(default_factory=dict)


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
        timings: bool = True,
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
        self.timings = timings  # log how long each stage took
        self._taken: set[Path] = set()

    @property
    def staging(self) -> Path:
        return self.library / ".staging"

    # ---- the whole source, stages overlapping ----

    def sync(self, source: AudioSource, retry: bool = False) -> list[SyncResult]:
        """Sync every file of `source`; `retry` also retries files that gave up."""
        files = list(source.list_files())
        results: list[SyncResult] = []
        new: list[int] = []
        compressing: list[tuple[_Job, int, Future]] = []
        with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as packer:

            def plan(i: int) -> _Job | None:
                if i >= len(files):
                    return None
                job = self._plan(files[i], retry)
                if job.result is None:
                    job.fetch = reader.submit(self._fetch, job)
                return job

            nxt = plan(0)
            for i in range(len(files)):
                job, nxt = nxt, plan(i + 1)  # start reading the next file now
                if job.fetch is not None:
                    job.fetch.result()
                r = self._process(job, retry)
                if r.status == "new" and job.dest is not None:
                    fut = packer.submit(self._pack, job)
                    compressing.append((job, r.recording_id, fut))
                elif r.status == "new":
                    self._cleanup(job.f.path, str(job.copy))
                results.append(r)
                if r.status == "new":
                    new.append(r.recording_id)
                if r.status != "known":
                    why = f" ({r.error})" if r.error else ""
                    self.log(f"{job.f.path.name}: {r.status}{why}")
                compressing = self._finish_packed(compressing)
            compressing = self._finish_packed(compressing, wait=True)
        self._tidy_staging()
        for rid in new:
            self.run_after(rid)
        return results

    def sync_file(self, f: AudioFile, retry: bool = False, after: bool = True) -> SyncResult:
        """Sync one file, start to finish (compression included). With after=False the
        follow-up work is left to the caller (run_after), e.g. until a batch is in."""
        job = self._plan(f, retry)
        if job.result is None:
            self._fetch(job)
        r = self._process(job, retry)
        if r.status == "new":
            if job.dest is not None:
                self._pack(job)
                self._finish_packed([(job, r.recording_id, None)], wait=True)
            else:
                self._cleanup(f.path, str(job.copy))
            self._tidy_staging()
            if after:
                self.run_after(r.recording_id)
        return r

    # ---- the stages ----

    def _plan(self, f: AudioFile, retry: bool) -> _Job:
        """What is known about a file before reading it (main thread)."""
        job = _Job(f)
        try:
            st = os.stat(f.path)
            job.size, job.mtime = st.st_size, st.st_mtime
        except OSError as e:  # unreadable on the source: nothing to record against
            job.result = SyncResult(f.path, "failed", error=f"{type(e).__name__}: {e}")
            return job
        seen = self.store.file_by_source(str(f.path), job.size, job.mtime)
        if seen is None:
            return job
        job.digest, job.stat_known = seen["sha256"], True
        decided = self._by_digest(job, retry, seen)
        if decided is not None:
            job.result = decided
        elif seen["path"] and Path(seen["path"]).is_file():
            job.copy = Path(seen["path"])  # a copy left by a failed attempt
        return job

    def _fetch(self, job: _Job) -> None:
        """Copy the file to the library (or to staging, if it is to be compressed)
        while hashing it, unless a usable copy exists already (reader thread)."""
        f = job.f
        compress = self.compress and f.path.suffix.lower() == ".wav"
        try:
            if job.copy is not None:
                if compress and job.copy.parent == self.staging:
                    job.dest = self._reserve(f, True)
                return
            dest = self._reserve(f, compress)
            target = (self.staging / dest.name).with_suffix(".wav") if compress else dest
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            try:
                digest = copy_and_hash(f.path, tmp)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
            if job.digest is None:
                job.digest = digest
            job.copy, job.fresh = target, True
            job.dest = dest if compress else None
            if not compress:
                self._taken.discard(dest)
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"

    def _reserve(self, f: AudioFile, compress: bool) -> Path:
        dest = library_path(self.library, f, compress, self._taken)
        self._taken.add(dest)
        return dest

    def _by_digest(self, job: _Job, retry: bool, status: dict | None) -> SyncResult | None:
        """Decide a file by its content, if that is enough (main thread)."""
        f = job.f
        known = self.store.find_by_hash(job.digest)
        if known and Path(known.path).parent == self.staging and Path(known.path).is_file():
            # an earlier sync stopped before compressing it: finish that now
            job.copy, job.dest = Path(known.path), self._reserve(f, True)
            self._pack(job)
            self._finish_packed([(job, known.id, None)], wait=True)
            return SyncResult(f.path, "known", known.id)
        if known:
            self._record(job, "done", recording_id=known.id)
            self._cleanup(f.path, known.path)
            return SyncResult(f.path, "known", known.id)
        if status and status["status"] == "noise":
            self._drop(f.path)
            return SyncResult(f.path, "noise")
        if status and status["status"] == "failed" and not retry:
            if status["attempts"] >= self.max_attempts:
                return SyncResult(f.path, "gave-up", error=status["error"])
        return None

    def _record(self, job: _Job, status: str, **fields) -> None:
        self.store.set_file_status(
            job.digest, str(job.f.path), status, size=job.size, mtime=job.mtime, **fields
        )

    def _discard(self, job: _Job) -> None:
        """Remove a copy this sync made that turned out not to be needed."""
        if job.fresh and job.copy is not None:
            job.copy.unlink(missing_ok=True)
        if job.dest is not None:
            self._taken.discard(job.dest)

    def _process(self, job: _Job, retry: bool) -> SyncResult:
        """Transcribe a fetched file and record the outcome (main thread)."""
        f = job.f
        if job.result is not None:
            return job.result
        if job.error is not None:
            if job.digest is None:
                return SyncResult(f.path, "failed", error=job.error)
            self._record(job, "failed", error=job.error)
            return SyncResult(f.path, "failed", error=job.error)
        if not job.stat_known:
            status = self.store.file_status(job.digest)
            decided = self._by_digest(job, retry, status)
            if decided is not None:
                self._discard(job)
                return decided
            if status and status["path"] and Path(status["path"]).is_file() and job.fresh:
                # a copy from an earlier attempt exists: use it, drop the new one
                self._discard(job)
                job.copy, job.fresh = Path(status["path"]), False
                if self.compress and job.copy.parent == self.staging:
                    job.dest = self._reserve(f, True)
                else:
                    job.dest = None
        try:
            self._record(job, "pending", path=str(job.copy), started_at=f.started_at)
            r = self.pipeline.process_file(job.copy, f.started_at, digest=job.digest)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self._record(job, "failed", error=error)
            if job.dest is not None:
                self._taken.discard(job.dest)
            return SyncResult(f.path, "failed", error=error)
        if self.timings and r.timings:
            self.log(f"{f.path.name}: {r.timing()}")
        if self.drop_noise and not has_speech(self.store.segments(r.recording_id)):
            self.store.delete(r.recording_id)
            job.copy.unlink(missing_ok=True)
            if job.dest is not None:
                self._taken.discard(job.dest)
            self._record(job, "noise", path=None)
            self._drop(f.path)
            return SyncResult(f.path, "noise")
        self._record(job, "done", recording_id=r.recording_id)
        return SyncResult(f.path, "new", r.recording_id)

    def _pack(self, job: _Job) -> None:
        """Compress the staged WAV into the library (packer thread). On failure the
        WAV itself goes into the library instead."""
        tmp = job.dest.with_name(job.dest.name + ".part")
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            compress_to_opus(job.copy, tmp)
            tmp.replace(job.dest)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            job.extra["pack_error"] = f"{type(e).__name__}: {e}"
            job.dest = job.dest.with_suffix(".wav")
            shutil.move(str(job.copy), job.dest)

    def _finish_packed(self, packing: list, wait: bool = False) -> list:
        """Point recordings at their compressed copies once ready (main thread);
        returns the ones still being compressed."""
        left = []
        for job, rid, fut in packing:
            if fut is not None and not fut.done() and not wait:
                left.append((job, rid, fut))
                continue
            if fut is not None:
                fut.result()
            if "pack_error" in job.extra:
                self.log(f"{job.f.path.name}: kept as WAV ({job.extra['pack_error']})")
            self.store.set_path(rid, job.dest)
            self._record(job, "done", path=str(job.dest), recording_id=rid)
            if job.copy.exists() and job.copy != job.dest:
                job.copy.unlink()
            self._taken.discard(job.dest)
            self._cleanup(job.f.path, str(job.dest))
        return left

    def _tidy_staging(self) -> None:
        try:
            self.staging.rmdir()  # only if empty: failed files keep their copies there
        except OSError:
            pass

    def run_after(self, rid: int) -> None:
        """The follow-up work for a new recording (`after`); a failure is only logged."""
        if self.after:
            try:
                self.after(rid)
            except Exception as e:  # extras (suggestions) can be redone later
                self.log(f"recording {rid}: after-sync step failed ({type(e).__name__}: {e})")

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
