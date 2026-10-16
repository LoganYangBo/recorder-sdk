"""Local web UI: `recorder ui` serves a single page plus a small JSON API on
127.0.0.1. Standard library only; the speaker model loads on first use.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import ops
from .store import Store

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


class App:
    """State shared by request threads. SQLite connections are per request;
    the speaker model is loaded once and used under a lock.
    """

    def __init__(self, db: str | Path, device: str = "auto", embedder=None):
        self.db = str(db)
        self.device = device
        self._embedder = embedder
        self.model_lock = threading.Lock()

    def store(self) -> Store:
        return Store(self.db)

    def embedder(self):
        if self._embedder is None:
            from .speaker import EcapaEmbedder

            self._embedder = EcapaEmbedder(device=self.device)
        return self._embedder

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
                    "name": Path(r.path).name,
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
            return [
                {
                    "n": i,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                    "score": s.speaker_score,
                    "match": s.speaker_match,
                    "logprob": s.avg_logprob,
                    "words": [[w.word, w.start, w.end, w.probability] for w in s.words],
                }
                for i, s in enumerate(store.segments(rid, words=True), 1)
            ]

    def search(self, q, body):
        query = q.get("q", [""])[0].strip()
        if not query:
            return []
        with self.store() as store:
            return [
                {
                    "recording": rec.id,
                    "name": Path(rec.path).name,
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
        with self.store() as store, self.model_lock:
            try:
                audios = ops.segment_audio(store, int(rid), indices)
                ops.enroll_voice(store, self.embedder(), name, audios)
            except (ValueError, TypeError) as e:
                raise ApiError(400, str(e)) from e
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
        return {"ok": True}

    def forget(self, q, body, name: str):
        with self.store() as store:
            if not store.delete_voiceprint(name):
                raise ApiError(404, f"no voiceprint '{name}'")
        return {"ok": True}

    def relabel(self, q, body):
        threshold = float(body.get("threshold", 0.3))
        rescore = bool(body.get("rescore", False))
        with self.store() as store, self.model_lock:
            try:
                labeler = ops.voiceprint_labeler(store, self.embedder(), threshold)
            except ValueError as e:
                raise ApiError(400, str(e)) from e
            if labeler is None:
                raise ApiError(400, "no voiceprints enrolled yet")
            ids = body.get("recordings") or [r.id for r in store.recordings()]
            try:
                ops.relabel(store, labeler, [int(i) for i in ids], rescore)
            except FileNotFoundError as e:
                raise ApiError(400, f"audio file not found: {e.filename}") from e
        return {"ok": True, "recordings": len(ids)}

    def audio_path(self, rid: int) -> Path:
        with self.store() as store:
            rec = store.recording(rid)
        if rec is None:
            raise ApiError(404, f"no recording {rid}")
        path = Path(rec.path)
        if not path.is_file():
            raise ApiError(404, f"audio file not found: {rec.path}")
        return path


ROUTES = [
    ("GET", r"/api/recordings", "recordings"),
    ("GET", r"/api/recordings/(\d+)/segments", "segments"),
    ("GET", r"/api/search", "search"),
    ("GET", r"/api/speakers", "speakers"),
    ("POST", r"/api/speakers/enroll", "enroll"),
    ("DELETE", r"/api/speakers/([^/]+)", "forget"),
    ("POST", r"/api/relabel", "relabel"),
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
            m = re.fullmatch(r"/api/recordings/(\d+)/audio", url.path)
            if self.command in ("GET", "HEAD") and m:
                return self._audio(app.audio_path(int(m.group(1))))
            for method, pattern, name in ROUTES:
                m = re.fullmatch(pattern, url.path)
                if m and method == self.command:
                    body = {}
                    if self.command == "POST":
                        n = int(self.headers.get("Content-Length") or 0)
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
) -> None:
    app = App(db, device=device)
    Store(db).close()  # create / migrate the database before the first request
    server = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"recorder UI on {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
