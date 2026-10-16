import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from recorder import Segment, Store, Transcript, Word
from recorder import speaker as speaker_mod
from recorder.web import App, make_handler

SR = speaker_mod.SAMPLE_RATE


def tone(freq, seconds):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class BandEmbedder:
    name = "speechbrain/spkrec-ecapa-voxceleb"

    def embed(self, audio):
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / SR)
        return np.array([spec[freqs < 500].sum(), spec[freqs >= 500].sum()], np.float32)


@pytest.fixture
def server(tmp_path, monkeypatch):
    audio_file = tmp_path / "talk.m4a"
    audio_file.write_bytes(bytes(range(256)) * 4)  # 1024 fake bytes
    audio = np.concatenate([tone(200, 2), tone(800, 2), tone(800, 2)])
    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: audio)
    db = tmp_path / "t.db"
    with Store(db) as store:
        t = Transcript(
            "en",
            6,
            [
                Segment(
                    0, 2, "hello Logan", words=[Word(0, 1, "hello", 0.9), Word(1, 2, "Logan", 0.2)]
                ),
                Segment(2, 4, "hi there"),
                Segment(4, 6, "how are you"),
            ],
            params={"model": "large-v3"},
        )
        store.save(audio_file, "h", t)
        store.save_voiceprint("me", BandEmbedder.name, np.array([1, 0], np.float32).tobytes())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(db, embedder=BandEmbedder())))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(base, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as r:
        raw = r.read()
        return r.status, dict(r.headers), raw


def jcall(base, path, method="GET", body=None):
    return json.loads(call(base, path, method, body)[2])


def test_index_and_listing(server):
    status, headers, html = call(server, "/")
    assert status == 200 and b"<title>Recorder</title>" in html
    (rec,) = jcall(server, "/api/recordings")
    assert rec["name"] == "talk.m4a" and rec["segments"] == 3 and rec["model"] == "large-v3"
    segs = jcall(server, "/api/recordings/1/segments")
    assert [s["n"] for s in segs] == [1, 2, 3]
    assert segs[0]["words"][1] == ["Logan", 1, 2, 0.2]
    assert jcall(server, "/api/search?q=there")[0]["text"] == "hi there"


def test_audio_supports_ranges(server):
    status, headers, body = call(server, "/api/recordings/1/audio")
    assert status == 200 and len(body) == 1024 and headers["Content-Type"] == "audio/mp4"
    status, headers, body = call(
        server, "/api/recordings/1/audio", headers={"Range": "bytes=10-19"}
    )
    assert status == 206 and body == bytes(range(10, 20))
    assert headers["Content-Range"] == "bytes 10-19/1024"


def test_enroll_relabel_forget(server):
    assert jcall(server, "/api/relabel", "POST", {"threshold": 0.8, "rescore": True})["ok"]
    assert [s["speaker"] for s in jcall(server, "/api/recordings/1/segments")] == [
        "me",
        "other",
        "other",
    ]

    jcall(
        server,
        "/api/speakers/enroll",
        "POST",
        {"name": "Alice", "recording": 1, "segments": [2, 3]},
    )
    jcall(server, "/api/relabel", "POST", {"threshold": 0.8, "rescore": True})
    segs = jcall(server, "/api/recordings/1/segments")
    assert [s["speaker"] for s in segs] == ["me", "Alice", "Alice"]
    speakers = jcall(server, "/api/speakers")
    assert [(s["name"], s["segments"]) for s in speakers] == [("me", 1), ("Alice", 2)]

    jcall(server, "/api/relabel", "POST", {"threshold": 1.01})  # threshold only
    assert {s["speaker"] for s in jcall(server, "/api/recordings/1/segments")} == {"other"}

    jcall(server, "/api/speakers/Alice", "DELETE")
    assert [s["name"] for s in jcall(server, "/api/speakers")] == ["me"]


def test_errors_are_json(server):
    for path, method, body, status in [
        ("/api/recordings/9/segments", "GET", None, 404),
        ("/api/speakers/enroll", "POST", {"name": "other", "recording": 1, "segments": [1]}, 400),
        ("/api/speakers/enroll", "POST", {"name": "x", "recording": 1, "segments": [7]}, 400),
        ("/api/nope", "GET", None, 404),
    ]:
        with pytest.raises(urllib.error.HTTPError) as e:
            call(server, path, method, body)
        assert e.value.code == status and "error" in json.loads(e.value.read())
