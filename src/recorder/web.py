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
    ):
        self.db = str(db)
        self.pronouncer = pronouncer
        self._audio_cache: tuple[str, object] | None = None  # (path, samples) of the last file
        self._no_pronunciation: set[str] = set()
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

    def delete(self, q, body, rid: int):
        """Delete a recording. Its audio file is deleted too when it lives in the
        library folder (imports and live recordings); files elsewhere are kept.
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
        job = {
            "id": next(self._job_ids),
            "name": path.name,
            "status": "queued",
            "recording": None,
            "error": None,
        }
        self.jobs[job["id"]] = job
        self._queue.put((job, path, threshold))
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._worker.start()
        return job

    def _work(self):
        while True:
            job, path, threshold = self._queue.get()
            job["status"] = "running"
            try:
                with self.model_lock, self.store() as store:
                    labeler = self._labeler(store, threshold)
                    r = Pipeline(self.transcriber(), store, labeler).process_file(path)
                    if r.skipped:  # same audio was imported before: drop the copy
                        path.unlink(missing_ok=True)
                        job["name"] = store.recording(r.recording_id).name
                job.update(status="duplicate" if r.skipped else "done", recording=r.recording_id)
            except Exception as e:
                job.update(status="error", error=f"{type(e).__name__}: {e}")

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
        if session.duration < 1:
            return {"recording": None, "committed": committed}
        with self.store() as store:
            params = getattr(self.transcriber(), "params", {})
            rid = session.save(store, self.library, params)
        return {"recording": rid, "committed": committed}

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
        w = ctx["word"]
        heard = w.original if w.original is not None else w.word
        alts, rank = [], None
        with self.model_lock:
            try:
                audio = self._recording_audio(rec.path)
                t0 = ctx["segment_start"]
                clip = audio[int(t0 * 16000) : int(ctx["segment_end"] * 16000)]
                tr = self.transcriber()
                alts = tr.alternatives(clip, w.start - t0, w.end - t0, language=rec.language)
                rank = tr.word_rank
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
            cands = learn.candidates(heard, alts, rank)
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
            word = store.word_context(wid)["word"].word  # as shown, e.g. "Think,"
        return {**c, "word": word, "compare": learn.compare(c["chosen"], c["heard"])}

    def word_compare(self, q, body):
        from . import learn

        target, heard = q.get("target", [""])[0], q.get("heard", [""])[0]
        return learn.compare(target, heard)

    def correction_list(self, q, body):
        from . import learn

        with self.store() as store:
            rows = [c for c in store.corrections() if c["heard"] != c["chosen"]]
        return [{**c, "compare": learn.compare(c["chosen"], c["heard"])} for c in rows]

    def corrections_markdown(self) -> bytes:
        from . import learn

        with self.store() as store:
            rows = [c for c in store.corrections() if c["heard"] != c["chosen"]]
        lines = [
            "# 纠错本",
            "",
            "| 时间 | 应为 | 听成 | 音标 | 句子 | 录音 |",
            "|---|---|---|---|---|---|",
        ]
        for c in rows:
            cmp = learn.compare(c["chosen"], c["heard"])
            sounds = f"/{cmp['target_ipa']}/ → /{cmp['heard_ipa']}/" if cmp["target_ipa"] else ""
            m, sec = divmod(int(c["start"]), 60)
            cells = [
                c["created_at"],
                c["chosen"],
                c["heard"],
                sounds,
                c["sentence"],
                f"{c['recording'] or '(已删除)'} @ {m}:{sec:02d}",
            ]
            lines.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
        return ("\n".join(lines) + "\n").encode()

    def pronunciation(self, word: str) -> bytes:
        from . import learn

        word = learn.normalize(word)
        if not word or not re.fullmatch(r"[a-z']+", word):
            raise ApiError(400, "not a word")
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
    ("DELETE", r"/api/recordings/(\d+)", "delete"),
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
                    app.corrections_markdown(),
                    "text/markdown; charset=utf-8",
                    {"Content-Disposition": "attachment; filename=corrections.md"},
                )
            if self.command == "GET" and url.path == "/api/pronounce":
                word = parse_qs(url.query).get("word", [""])[0]
                return self._send(
                    200, app.pronunciation(word), "audio/mpeg", {"Cache-Control": "max-age=86400"}
                )
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


def serve(
    db: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    device: str = "auto",
    open_browser: bool = True,
    library: str | Path | None = None,
    whisper: dict | None = None,
) -> None:
    app = App(db, device=device, library=library, whisper=whisper)
    Store(db).close()  # create / migrate the database before the first request
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
