"""Local web UI: `recorder ui` serves a single page plus a small JSON API on
127.0.0.1. Standard library only; the speaker model loads on first use.
"""

from __future__ import annotations

import itertools
import json
import mimetypes
import queue
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import ops
from .ingest import has_speech
from .live import LiveSession
from .pipeline import Pipeline
from .store import Store
from .sync import AUDIO_EXTS

AUDIO_TYPES = {
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}


# the exported notebook, in the web UI's language
MARKDOWN_TEXT = {
    "en": {
        "title": "Pronunciation notebook",
        "columns": ["Time", "Meant", "Heard", "IPA", "Sentence", "Recording"],
        "confirmed": "(right, but Whisper was unsure)",
        "deleted": "(deleted)",
    },
    "zh": {
        "title": "纠错本",
        "columns": ["时间", "应为", "听成", "音标", "句子", "录音"],
        "confirmed": "（识别对了，但没把握）",
        "deleted": "（已删除）",
    },
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class RawBody:
    """Request body for routes that take audio bytes instead of JSON."""

    def __init__(self, rfile, length: int):
        self.rfile, self.length = rfile, length

    def read_all(self) -> bytes:
        return self.rfile.read(self.length)

    def copy_to(self, path: Path) -> None:
        left = self.length
        with open(path, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    raise ApiError(400, "upload was cut off")
                f.write(chunk)
                left -= len(chunk)


def _safe_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name  # drop any directory part
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "upload"


def _seg_json(n: int, s) -> dict:
    return {
        "n": n,
        "start": s.start,
        "end": s.end,
        "text": s.text,
        "speaker": s.speaker,
        "score": s.speaker_score,
        "match": s.speaker_match,
        "logprob": s.avg_logprob,
        # [word, start, end, probability, id, original (if corrected/confirmed)]
        "words": [[w.word, w.start, w.end, w.probability, w.id, w.original] for w in s.words],
    }


DICTIONARY_API = "https://api.dictionaryapi.dev/api/v2/entries/en/"


def fetch_pronunciation(word: str) -> bytes | None:
    """A human recording of `word` (US if there is one) from the free
    dictionaryapi.dev (Wiktionary audio), or None."""
    import urllib.error
    import urllib.request
    from urllib.parse import quote

    try:
        with urllib.request.urlopen(DICTIONARY_API + quote(word), timeout=6) as r:
            entries = json.loads(r.read())
        urls = [
            ph["audio"]
            for e in entries
            for ph in e.get("phonetics", [])
            if ph.get("audio", "").startswith("http")
        ]
        if not urls:
            return None
        urls.sort(key=lambda u: (not u.endswith("-us.mp3"), u))
        with urllib.request.urlopen(urls[0], timeout=10) as r:
            return r.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None


class App:
    """State shared by request threads. SQLite connections are per request.
    Whisper and the speaker model are loaded once, on first use, and only
    used under `model_lock` (one GPU job at a time). Imports are transcribed
    one by one on a background thread.
    """

    def __init__(
        self,
        db: str | Path,
        device: str = "auto",
        embedder=None,
        transcriber=None,
        library: str | Path | None = None,
        whisper: dict | None = None,
        pronouncer=fetch_pronunciation,
        aligner=None,
        link=None,
    ):
        self.db = str(db)
        # forced aligner for precise word times: an object, "auto" (load on first use), or None
        self._aligner = aligner
        self.pronouncer = pronouncer
        self._audio_cache: tuple[str, object] | None = None  # (path, samples) of the last file
        self._no_pronunciation: set[str] = set()
        self._voices = None  # tts.KokoroVoices, loaded on first use
        self.device = device
        self.library = Path(library) if library else Path(db).resolve().parent / "recordings"
        self.whisper = whisper or {}
        self._embedder = embedder
        self._transcriber = transcriber
        self.model_lock = threading.RLock()
        self.jobs: dict[int, dict] = {}
        self._job_ids = itertools.count(1)
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self.live: dict[int, LiveSession] = {}
        self._live_ids = itertools.count(1)
        # the recorder over Bluetooth (ble.BleLink, made on first use) and what it last said
        self._link = link
        self.gadget = {"status": None, "files": None, "error": None}
        self.gadget_live: dict | None = None  # {"session", "queue", "level", "segs"} while on

    def store(self) -> Store:
        return Store(self.db)

    def embedder(self):
        if self._embedder is None:
            from .speaker import EcapaEmbedder

            self._embedder = EcapaEmbedder(device=self.device)
        return self._embedder

    def transcriber(self):
        if self._transcriber is None:
            from .transcribe import WhisperTranscriber

            self._transcriber = WhisperTranscriber(device=self.device, **self.whisper)
        return self._transcriber

    def aligner(self):
        """The word aligner, or None if off or it can't be loaded (e.g. no
        torchaudio, or the model download failed)."""
        if self._aligner == "auto":
            try:
                from .align import WordAligner

                self._aligner = WordAligner(device=self.device)
            except Exception as e:
                print(f"word alignment off: {type(e).__name__}: {e}")
                self._aligner = None
        return self._aligner

    def _labeler(self, store: Store, threshold: float):
        try:
            return ops.voiceprint_labeler(store, self.embedder(), threshold)
        except ValueError as e:
            raise ApiError(400, str(e)) from e

    # ---- API handlers: (params, body) -> JSON-serialisable ----

    def recordings(self, q, body):
        with self.store() as store:
            counts = dict(
                store.conn.execute(
                    "SELECT recording_id, COUNT(*) FROM segments GROUP BY recording_id"
                ).fetchall()
            )
            return [
                {
                    "id": r.id,
                    "path": r.path,
                    "name": r.name,
                    "file": Path(r.path).name,
                    "title": r.title,
                    "duration": r.duration,
                    "language": r.language,
                    "started_at": r.started_at,
                    "model": r.params.get("model"),
                    "segments": counts.get(r.id, 0),
                }
                for r in store.recordings()
            ]

    def segments(self, q, body, rid: int):
        with self.store() as store:
            if store.recording(rid) is None:
                raise ApiError(404, f"no recording {rid}")
            return [_seg_json(i, s) for i, s in enumerate(store.segments(rid, words=True), 1)]

    def search(self, q, body):
        query = q.get("q", [""])[0].strip()
        if not query:
            return []
        with self.store() as store:
            return [
                {
                    "recording": rec.id,
                    "name": rec.name,
                    "started_at": rec.started_at,
                    "start": seg.start,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for rec, seg in store.search(query, limit=100)
            ]

    def speakers(self, q, body):
        with self.store() as store:
            rows = store.conn.execute(
                "SELECT v.name, v.created_at, "
                "  (SELECT COUNT(*) FROM segments s WHERE s.speaker = v.name) "
                "FROM voiceprints v ORDER BY v.name = 'me' DESC, v.name"
            ).fetchall()
        return [{"name": n, "created_at": c, "segments": k} for n, c, k in rows]

    def enroll(self, q, body):
        name = str(body.get("name", "")).strip()
        rid = body.get("recording")
        indices = [int(i) for i in body.get("segments", [])]
        threshold = float(body.get("threshold", 0.3))
        with self.store() as store, self.model_lock:
            try:
                audios = ops.segment_audio(store, int(rid), indices)
                ops.enroll_voice(store, self.embedder(), name, audios)
                # re-label this recording right away so the new name shows up;
                # other recordings are re-scored on demand (it re-reads their audio)
                ops.relabel(store, self._labeler(store, threshold), [int(rid)], rescore=True)
            except (ValueError, TypeError) as e:
                raise ApiError(400, str(e)) from e
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
            labelled = sum(s.speaker == name for s in store.segments(int(rid)))
        return {"ok": True, "labelled": labelled}

    def forget(self, q, body, name: str):
        with self.store() as store:
            if not store.delete_voiceprint(name):
                raise ApiError(404, f"no voiceprint '{name}'")
        return {"ok": True}

    def relabel(self, q, body):
        threshold = float(body.get("threshold", 0.3))
        rescore = bool(body.get("rescore", False))
        with self.store() as store, self.model_lock:
            labeler = self._labeler(store, threshold)
            if labeler is None:
                raise ApiError(400, "no voiceprints enrolled yet")
            ids = body.get("recordings") or [r.id for r in store.recordings()]
            try:
                ops.relabel(store, labeler, [int(i) for i in ids], rescore)
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
        return {"ok": True, "recordings": len(ids)}

    def rename(self, q, body, rid: int):
        title = str(body.get("title") or "")[:200]
        with self.store() as store:
            if not store.rename(rid, title):
                raise ApiError(404, f"no recording {rid}")
            return {"ok": True, "name": store.recording(rid).name}

    def set_started(self, q, body, rid: int):
        """Set (or clear, with "") when a recording started: local time, ISO 8601."""
        from datetime import datetime

        value = str(body.get("started_at") or "").strip()
        if value:
            try:
                value = (
                    datetime.fromisoformat(value).replace(tzinfo=None, microsecond=0).isoformat()
                )
            except ValueError as e:
                raise ApiError(400, f"not a date and time: {value}") from e
        with self.store() as store:
            if store.recording(rid) is None:
                raise ApiError(404, f"no recording {rid}")
            store.set_started_at(rid, value or None)
        return {"ok": True, "started_at": value or None}

    def delete(self, q, body, rid: int):
        """Delete a recording. Its audio file is deleted too when it lives in the
        library folder (imports, live recordings, synced files); files elsewhere are kept.
        """
        with self.store() as store:
            rec = store.recording(rid)
            if rec is None:
                raise ApiError(404, f"no recording {rid}")
            path = Path(rec.path).resolve()
            in_library = path.is_relative_to(self.library.resolve())
            store.delete(rid)
        deleted_file = False
        if in_library and path.is_file():
            path.unlink()
            deleted_file = True
        return {"ok": True, "deleted_file": deleted_file, "path": rec.path}

    def delete_many(self, q, body):
        """Delete several recordings (body: {"ids": [...]}), each as `delete` does."""
        deleted, kept = 0, []
        for rid in body.get("ids", []):
            try:
                r = self.delete(q, {}, int(rid))
            except ApiError:  # already gone
                continue
            deleted += 1
            if not r["deleted_file"]:
                kept.append(r["path"])
        return {"deleted": deleted, "kept_files": kept}

    # ---- import: upload into the library, transcribe in the background ----

    def upload(self, q, body: RawBody):
        name = _safe_name(q.get("name", ["upload"])[0])
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            raise ApiError(400, f"not an audio file: {name} ({', '.join(sorted(AUDIO_EXTS))})")
        self.library.mkdir(parents=True, exist_ok=True)
        path = self.library / name
        stem, suffix = path.stem, path.suffix
        for i in itertools.count(1):
            if not path.exists():
                break
            path = self.library / f"{stem}-{i}{suffix}"
        body.copy_to(path)
        threshold = float(q.get("threshold", ["0.3"])[0])
        return self._enqueue("import", path.name, lambda job: self._import(job, path, threshold))

    def _enqueue(self, kind: str, name: str, run, recording: int | None = None) -> dict:
        job = {
            "id": next(self._job_ids),
            "kind": kind,
            "name": name,
            "status": "queued",
            "recording": recording,
            "progress": None,
            "error": None,
        }
        self.jobs[job["id"]] = job
        self._queue.put((job, run))
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()
        return job

    def _work(self):
        while True:
            job, run = self._queue.get()
            job["status"] = "running"
            try:
                run(job)
                job["status"] = job.get("result", "done")
            except Exception as e:
                job.update(status="error", error=f"{type(e).__name__}: {e}")

    def _import(self, job: dict, path: Path, threshold: float) -> None:
        with self.model_lock, self.store() as store:
            labeler = self._labeler(store, threshold)
            from .sync import started_at_of

            r = Pipeline(self.transcriber(), store, labeler).process_file(path, started_at_of(path))
            if r.skipped:  # same audio was imported before: drop the copy
                path.unlink(missing_ok=True)
                job["name"] = store.recording(r.recording_id).name
                result = "duplicate"
            elif not has_speech(store.segments(r.recording_id)):  # no speech: drop it
                store.delete(r.recording_id)
                path.unlink(missing_ok=True)
                job.update(result="noise")
                return
            else:
                result = "done"
        job.update(result=result, recording=r.recording_id)
        if result == "done":
            self.suggest(r.recording_id)

    def suggest(self, rid: int) -> dict | None:
        """Queue working out candidates for a recording's unsure words (None
        if there are none left to do)."""
        with self.store() as store:
            rec = store.recording(rid)
            if rec is None or not store.unsure_words(rid, align=self._aligner is not None):
                return None

        def run(job):
            def progress(done, total):
                job["progress"] = [done, total]

            with self.model_lock:
                aligner = self.aligner()
            with self.store() as store:
                ops.suggest_words(
                    store, self.transcriber(), rid, self.model_lock, progress, aligner
                )

        return self._enqueue("suggest", rec.name, run, recording=rid)

    def suggest_pending(self) -> list[dict]:
        """Queue every recording that still has unsure words not analysed."""
        with self.store() as store:
            pending = store.pending_unsure(align=self._aligner is not None)
        return [job for rid in pending if (job := self.suggest(rid))]

    def job_list(self, q, body):
        return sorted(self.jobs.values(), key=lambda j: -j["id"])[:20]

    # ---- live: microphone audio streamed from the page ----

    def live_start(self, q, body):
        threshold = float(body.get("threshold", 0.3))
        with self.store() as store:
            labeler = self._labeler(store, threshold)
        sid = next(self._live_ids)
        self.live[sid] = LiveSession(self.transcriber(), labeler)
        return {"id": sid}

    def _session(self, sid: int) -> LiveSession:
        if sid not in self.live:
            raise ApiError(404, f"no live session {sid}")
        return self.live[sid]

    def live_audio(self, q, body: RawBody, sid: int):
        import numpy as np

        session = self._session(sid)
        samples = np.frombuffer(body.read_all(), dtype="<f4")
        before = len(session.committed)
        with self.model_lock:
            session.feed(samples)
        return {
            "committed": [
                _seg_json(before + i + 1, s) for i, s in enumerate(session.committed[before:])
            ],
            "partial": session.partial,
            "duration": session.duration,
        }

    def live_stop(self, q, body, sid: int):
        session = self._session(sid)
        before = len(session.committed)
        with self.model_lock:
            session.finish()
        del self.live[sid]
        committed = [_seg_json(before + i + 1, s) for i, s in enumerate(session.committed[before:])]
        if session.duration < 1 or not has_speech(session.committed):  # nothing worth keeping
            reason = "short" if session.duration < 1 else "empty"
            return {"recording": None, "committed": committed, "reason": reason}
        with self.store() as store:
            params = getattr(self.transcriber(), "params", {})
            rid = session.save(store, self.library, params)
        self.suggest(rid)
        return {"recording": rid, "committed": committed}

    # ---- the recorder over Bluetooth ----

    def link(self):
        if self._link is None:
            try:
                from .ble import BleLink, import_bleak

                import_bleak()
            except ImportError as e:
                raise ApiError(400, 'needs bleak: pip install "recorder[device]"') from e
            self._link = BleLink()
        return self._link

    def _gadget_call(self, work, wait: bool = True):
        from .ble import RecorderError

        try:
            return self.link().call(work, wait)
        except (RecorderError, OSError) as e:
            self.gadget["error"] = str(e)
            raise ApiError(400, str(e)) from e

    def _refresh_files(self) -> None:
        files = self._gadget_call(lambda rec: rec.list_files())
        self.gadget["files"] = {"count": len(files), "bytes": sum(n for _, n in files)}

    def gadget_state(self, q, body):
        g = self.gadget
        connected = self._link is not None and self._link.connected
        if connected:
            try:
                status = self._gadget_call(lambda rec: rec.status(), wait=False)
                if status is not None:  # None: busy with a transfer, keep the last one
                    g["status"] = status
            except ApiError:
                pass
        live = self.gadget_live
        return {
            "connected": connected,
            "status": g["status"] if connected else None,
            "files": g["files"] if connected else None,
            "error": g["error"],
            "live": live is not None,
            "mtu": getattr(self._link.client, "mtu_size", None) if connected else None,
        }

    def gadget_connect(self, q, body):
        from .ble import RecorderError

        self.gadget["error"] = None
        try:
            clock = self.link().connect()
        except (RecorderError, OSError, TimeoutError) as e:
            self.gadget["error"] = str(e) or type(e).__name__
            raise ApiError(400, self.gadget["error"]) from e
        self._refresh_files()
        return {"clock": clock, **self.gadget_state(q, body)}

    def gadget_disconnect(self, q, body):
        if self.gadget_live:
            self.gadget_live_stop(q, {})
        if self._link is not None:
            self._link.disconnect()
        return self.gadget_state(q, body)

    def gadget_sync(self, q, body):
        delete = bool(body.get("delete"))
        threshold = float(body.get("threshold", 0.3))
        self.link()
        for job in self.jobs.values():  # one sync at a time: a second click joins the first
            if job["kind"] == "device" and job["status"] in ("queued", "running"):
                return job
        return self._enqueue("device", "Recorder", lambda job: self._sync(job, delete, threshold))

    def _sync(self, job: dict, delete: bool, threshold: float) -> None:
        """Download each recording, then import it before the next one, so results come in
        as they go; the job shows the file count, bytes and download speed."""
        import time

        from .ingest import Syncer
        from .sync import AudioFile, started_at_of

        inbox = Path(self.db).resolve().parent / "inbox"
        files = self._gadget_call(lambda rec: rec.list_files())
        total = sum(n for _, n in files)
        job.update(progress=[0, len(files)], bytes=[0, total], rate=None)
        done_bytes = fetched = 0
        started = time.monotonic()
        new = []
        syncer = None  # the models load when the first file is in, not before
        with self.store() as store:
            for i, (path, size) in enumerate(files):
                job["stage"] = "downloading"
                dest = inbox / path
                if not (dest.is_file() and dest.stat().st_size == size):
                    base, before = done_bytes, fetched

                    def progress(got, base=base, before=before):
                        job["bytes"] = [base + got, total]
                        job["rate"] = (before + got) / max(time.monotonic() - started, 0.1)

                    self._gadget_call(
                        lambda rec, p=path, n=size, d=dest: rec.fetch(p, n, d, progress)
                    )
                    fetched += size
                done_bytes += size
                job["bytes"] = [done_bytes, total]
                job["stage"] = "transcribing"
                with self.model_lock:
                    if syncer is None:
                        labeler = self._labeler(store, threshold)
                        pipeline = Pipeline(self.transcriber(), store, labeler)
                        syncer = Syncer(store, pipeline, self.library, drop_noise=True)
                    r = syncer.sync_file(AudioFile(dest, started_at_of(dest)))
                if r.status == "new":
                    new.append(r.recording_id)
                    job.update(new=len(new), recording=r.recording_id)
                if r.status == "noise":
                    job["noise"] = job.get("noise", 0) + 1
                if r.status == "failed":
                    job["error"] = r.error
                if delete and r.status in ("new", "known", "noise"):
                    self._gadget_call(lambda rec, p=path: rec.delete(p))
                    dest.unlink(missing_ok=True)
                job["progress"] = [i + 1, len(files)]
        job.update(new=len(new), stage=None)
        self._refresh_files()
        for rid in new:
            self.suggest(rid)

    def gadget_live_start(self, q, body):
        import numpy as np

        if self.gadget_live:
            return {"ok": True}
        threshold = float(body.get("threshold", 0.3))
        with self.store() as store:
            labeler = self._labeler(store, threshold)
        live = {
            "session": LiveSession(self.transcriber(), labeler),
            "queue": queue.Queue(),
            "level": 0.0,
        }

        def on_audio(samples):  # on the Bluetooth thread: keep it quick
            x = samples.astype(np.float32) / 32768
            live["level"] = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
            live["queue"].put(x)

        def feed():
            session, chunks = live["session"], live["queue"]
            while (c := chunks.get()) is not None:
                parts = [c]
                while not chunks.empty() and (c := chunks.get_nowait()) is not None:
                    parts.append(c)
                with self.model_lock:
                    session.feed(np.concatenate(parts))
                if c is None:
                    break

        live["thread"] = threading.Thread(target=feed, daemon=True)
        live["thread"].start()
        self.gadget_live = live
        try:
            self._gadget_call(lambda rec: rec.live(on_audio))
        except ApiError:
            self.gadget_live = None
            live["queue"].put(None)
            raise
        return {"ok": True}

    def gadget_live_poll(self, q, body):
        live = self.gadget_live
        if live is None:
            return {"on": False, "committed": [], "partial": "", "duration": 0, "level": 0}
        session = live["session"]
        start = int(q.get("from", ["0"])[0])
        segs, partial = session.committed[start:], session.partial  # no lock: feed holds it
        return {
            "on": True,
            "committed": [_seg_json(start + i + 1, s) for i, s in enumerate(segs)],
            "partial": partial,
            "duration": session.duration,
            "level": live["level"],
        }

    def gadget_live_stop(self, q, body):
        live, self.gadget_live = self.gadget_live, None
        if live is None:
            return {"recording": None, "committed": []}
        try:
            self._gadget_call(lambda rec: rec.live(None))
        except ApiError:
            pass  # gone already: the audio received is still transcribed
        live["queue"].put(None)
        live["thread"].join()
        session = live["session"]
        before = len(session.committed)
        with self.model_lock:
            session.finish()
        committed = [_seg_json(before + i + 1, s) for i, s in enumerate(session.committed[before:])]
        # not saved: the recorder puts the same audio on its card, which syncs in full
        return {"recording": None, "committed": committed}

    # ---- unsure words: candidates, corrections, pronunciation ----

    def _recording_audio(self, path: str):
        from .speaker import load_audio

        if self._audio_cache is None or self._audio_cache[0] != path:
            self._audio_cache = (path, load_audio(path))
        return self._audio_cache[1]

    def word_candidates(self, q, body, wid: int):
        from . import learn

        with self.store() as store:
            ctx = store.word_context(wid)
            if ctx is None:
                raise ApiError(404, f"no word {wid}")
            rec = store.recording(ctx["recording_id"])
            cands = store.candidates(wid)  # usually worked out in the background already
        w = ctx["word"]
        heard = w.original if w.original is not None else w.word
        if cands is None:
            with self.model_lock:
                try:
                    audio = self._recording_audio(rec.path)
                except FileNotFoundError as e:
                    raise ApiError(400, f"audio file not found: {e.filename}") from e
                t0 = ctx["segment_start"]
                clip = audio[int(t0 * 16000) : int(ctx["segment_end"] * 16000)]
                tr = self.transcriber()
                (alts,) = tr.alternatives(clip, [(w.start - t0, w.end - t0)], language=rec.language)
                cands = learn.candidates(heard, alts, tr.word_rank)
            with self.store() as store:
                store.set_candidates(wid, cands)
        return {
            "id": wid,
            "word": w.word,
            "heard": learn.normalize(heard),
            "probability": w.probability,
            "corrected": w.original is not None,
            "speaker": ctx["speaker"],
            "heard_ipa": learn.ipa(heard),
            "candidates": [{"word": c, "ipa": learn.ipa(c)} for c in cands],
        }

    def word_clip(self, wid: int, pad: float = 0.05) -> bytes:
        """Just this word's audio, as a WAV, cut at its (aligned) times with
        short fades; exact, unlike seeking in the compressed recording."""
        import io
        import wave

        import numpy as np

        with self.store() as store:
            ctx = store.word_context(wid)
            if ctx is None:
                raise ApiError(404, f"no word {wid}")
            path = store.recording(ctx["recording_id"]).path
        w = ctx["word"]
        with self.model_lock:  # the audio cache is shared
            try:
                audio = self._recording_audio(path)
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
        clip = audio[max(0, int((w.start - pad) * 16000)) : int((w.end + pad) * 16000)].copy()
        fade = min(160, len(clip) // 2)  # 10 ms
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes((np.clip(clip, -1, 1) * 32767).astype("<i2").tobytes())
        return buf.getvalue()

    def word_correct(self, q, body, wid: int):
        from . import learn

        chosen = learn.normalize(str(body.get("word", "")))
        if not chosen:
            raise ApiError(400, "empty word")
        with self.store() as store:
            try:
                c = store.correct_word(wid, chosen)
            except KeyError as e:
                raise ApiError(404, str(e)) from e
            w = store.word_context(wid)["word"]
        return {
            **c,
            "word": w.word,  # as shown, e.g. "Think,"
            "probability": w.probability,
            "compare": learn.compare(c["chosen"], c["heard"]),
        }

    def word_compare(self, q, body):
        from . import learn

        target, heard = q.get("target", [""])[0], q.get("heard", [""])[0]
        return learn.compare(target, heard)

    def correction_list(self, q, body):
        from . import learn

        with self.store() as store:
            rows = store.corrections()  # confirmed words too: Whisper wasn't sure of them
        return [{**c, "compare": learn.compare(c["chosen"], c["heard"])} for c in rows]

    def corrections_markdown(self, lang: str = "en") -> bytes:
        from . import learn

        text = MARKDOWN_TEXT.get(lang, MARKDOWN_TEXT["en"])
        with self.store() as store:
            rows = store.corrections()
        lines = [f"# {text['title']}", "", "| " + " | ".join(text["columns"]) + " |"]
        lines.append("|---" * len(text["columns"]) + "|")
        for c in rows:
            cmp = learn.compare(c["chosen"], c["heard"])
            same = c["heard"] == c["chosen"]
            sounds = f"/{cmp['target_ipa']}/" if same and cmp["target_ipa"] else ""
            if not same and cmp["target_ipa"]:
                sounds = f"/{cmp['target_ipa']}/ → /{cmp['heard_ipa']}/"
            m, sec = divmod(int(c["start"]), 60)
            cells = [
                c["created_at"],
                c["chosen"],
                text["confirmed"] if same else c["heard"],
                sounds,
                c["sentence"],
                f"{c['recording'] or text['deleted']} @ {m}:{sec:02d}",
            ]
            lines.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
        return ("\n".join(lines) + "\n").encode()

    def pronunciation(self, word: str, voice: str = "dict") -> tuple[bytes, str]:
        """(audio, content type) of `word` said by `voice`: dict (a human
        recording from the dictionary) or female / male (synthesised
        American English)."""
        from . import learn

        word = learn.normalize(word)
        if not word or not re.fullmatch(r"[a-z']+", word):
            raise ApiError(400, "not a word")
        if voice != "dict":
            return self._synthesised(word, voice), "audio/wav"
        return self._dictionary(word), "audio/mpeg"

    def _synthesised(self, word: str, voice: str) -> bytes:
        from . import tts

        if voice not in tts.VOICES:
            raise ApiError(400, f"unknown voice {voice}")
        cache = self.library.parent / "pronunciations" / voice / f"{word}.wav"
        if cache.is_file():
            return cache.read_bytes()
        with self.model_lock:
            try:
                if self._voices is None:
                    self._voices = tts.KokoroVoices(self.library.parent / "models" / "kokoro")
                samples, rate = self._voices.say(word, voice)
            except ImportError as e:
                raise ApiError(501, f'pip install -e ".[tts]" first ({e})') from e
        data = tts.to_wav(tts.trim(samples, rate), rate)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def _dictionary(self, word: str) -> bytes:
        cache = self.library.parent / "pronunciations" / f"{word}.mp3"
        if cache.is_file():
            return cache.read_bytes()
        if word in self._no_pronunciation:
            raise ApiError(404, f"no recording of '{word}'")
        data = self.pronouncer(word)
        if not data:
            self._no_pronunciation.add(word)
            raise ApiError(404, f"no recording of '{word}'")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def audio_path(self, rid: int) -> Path:
        with self.store() as store:
            rec = store.recording(rid)
        if rec is None:
            raise ApiError(404, f"no recording {rid}")
        path = Path(rec.path)
        if not path.is_file():
            raise ApiError(404, f"audio file not found: {rec.path}")
        return path


ROUTES = [  # (method, path, handler, raw body)
    ("GET", r"/api/recordings", "recordings"),
    ("POST", r"/api/recordings/(\d+)/rename", "rename"),
    ("POST", r"/api/recordings/(\d+)/started", "set_started"),
    ("DELETE", r"/api/recordings/(\d+)", "delete"),
    ("POST", r"/api/recordings/delete", "delete_many"),
    ("POST", r"/api/import", "upload", True),
    ("GET", r"/api/jobs", "job_list"),
    ("POST", r"/api/live/start", "live_start"),
    ("POST", r"/api/live/(\d+)/audio", "live_audio", True),
    ("POST", r"/api/live/(\d+)/stop", "live_stop"),
    ("GET", r"/api/recordings/(\d+)/segments", "segments"),
    ("GET", r"/api/search", "search"),
    ("GET", r"/api/speakers", "speakers"),
    ("POST", r"/api/speakers/enroll", "enroll"),
    ("DELETE", r"/api/speakers/([^/]+)", "forget"),
    ("POST", r"/api/relabel", "relabel"),
    ("GET", r"/api/words/(\d+)/candidates", "word_candidates"),
    ("POST", r"/api/words/(\d+)/correct", "word_correct"),
    ("GET", r"/api/compare", "word_compare"),
    ("GET", r"/api/corrections", "correction_list"),
    ("GET", r"/api/device", "gadget_state"),
    ("POST", r"/api/device/connect", "gadget_connect"),
    ("POST", r"/api/device/disconnect", "gadget_disconnect"),
    ("POST", r"/api/device/sync", "gadget_sync"),
    ("POST", r"/api/device/live/start", "gadget_live_start"),
    ("GET", r"/api/device/live", "gadget_live_poll"),
    ("POST", r"/api/device/live/stop", "gadget_live_stop"),
]


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, data):
            self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

        def _dispatch(self):
            url = urlparse(self.path)
            if self.command == "GET" and url.path in ("/", "/index.html"):
                html = resources.files("recorder").joinpath("web.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if self.command == "GET" and url.path == "/api/corrections.md":
                return self._send(
                    200,
                    app.corrections_markdown(parse_qs(url.query).get("lang", ["en"])[0]),
                    "text/markdown; charset=utf-8",
                    {"Content-Disposition": "attachment; filename=corrections.md"},
                )
            if self.command == "GET" and url.path == "/api/pronounce":
                q = parse_qs(url.query)
                data, ctype = app.pronunciation(q.get("word", [""])[0], q.get("voice", ["dict"])[0])
                return self._send(200, data, ctype, {"Cache-Control": "max-age=86400"})
            m = re.fullmatch(r"/api/words/(\d+)/clip.wav", url.path)
            if self.command == "GET" and m:
                return self._send(200, app.word_clip(int(m.group(1))), "audio/wav")
            m = re.fullmatch(r"/api/recordings/(\d+)/audio", url.path)
            if self.command in ("GET", "HEAD") and m:
                return self._audio(app.audio_path(int(m.group(1))))
            for method, pattern, name, *raw in ROUTES:
                m = re.fullmatch(pattern, url.path)
                if m and method == self.command:
                    body = {}
                    if self.command == "POST":
                        n = int(self.headers.get("Content-Length") or 0)
                        if raw:
                            body = RawBody(self.rfile, n)
                        else:
                            body = json.loads(self.rfile.read(n) or b"{}")
                    args = [int(g) if g.isdigit() else unquote(g) for g in m.groups()]
                    return self._json(200, getattr(app, name)(parse_qs(url.query), body, *args))
            raise ApiError(404, "not found")

        def _audio(self, path: Path):
            size = path.stat().st_size
            ctype = AUDIO_TYPES.get(path.suffix.lower()) or (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            start, end, status = 0, size - 1, 200
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
                status = 206
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if status == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._send(status, data, ctype, extra)

        def _handle(self):
            try:
                self._dispatch()
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # report instead of dropping the connection
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        do_GET = do_POST = do_DELETE = do_HEAD = _handle

    return Handler


def _fill_start_times(store: Store) -> None:
    """Recordings without a start time get one from their file name or metadata, if any;
    ones saved with a UTC offset (live recordings before 0.3) are put in local time."""
    from datetime import datetime

    from .sync import _local, started_at_of

    for r in store.recordings():
        if r.started_at and len(r.started_at) > 19:
            try:
                store.set_started_at(r.id, _local(datetime.fromisoformat(r.started_at)))
            except ValueError:
                pass
        elif not r.started_at and Path(r.path).is_file():
            found = started_at_of(r.path)
            if found:
                store.set_started_at(r.id, found)


def serve(
    db: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    device: str = "auto",
    open_browser: bool = True,
    library: str | Path | None = None,
    whisper: dict | None = None,
) -> None:
    import faulthandler
    from importlib.util import find_spec

    try:  # a crash in a native library (GPU, Bluetooth) then prints where it was
        faulthandler.enable()
    except (OSError, ValueError, AttributeError):  # stderr isn't a real file (tests)
        pass
    aligner = "auto" if find_spec("torchaudio") else None
    app = App(db, device=device, library=library, whisper=whisper, aligner=aligner)
    with Store(db) as store:  # create / migrate the database before the first request
        _fill_start_times(store)
    if app.suggest_pending():
        print("working out suggestions for unsure words in the background")
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"recorder UI on {url}  (Ctrl+C to stop)")
    print(f"imports and live recordings are saved in {app.library}")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
