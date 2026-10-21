import json
import threading
import time
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


class FakeTranscriber:
    params = {"model": "fake"}

    def transcribe(self, audio):
        if hasattr(audio, "dtype"):  # live: one segment per 2 s
            n = len(audio) // (2 * SR)
            segs = [Segment(2.0 * k, 2.0 * k + 2, f"live {k}") for k in range(n)]
            return Transcript("en", len(audio) / SR, segs)
        segs = [Segment(0, 2, "imported a"), Segment(2, 4, "imported b"), Segment(4, 6, "c")]
        return Transcript("en", 6, segs, params=self.params)

    calls = 0

    def alternatives(self, audio, spans, language=None):
        FakeTranscriber.calls += 1
        assert len(audio) == 2 * SR and spans == [(1, 2)]  # the word within its segment
        return [["Logan", "logging", "Logan", "logan", "lagoon"]]

    def word_rank(self, word):
        return 0


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
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            App(
                db,
                embedder=BandEmbedder(),
                transcriber=FakeTranscriber(),
                library=tmp_path / "lib",
                pronouncer=lambda w: b"MP3" if w == "logan" else None,
            )
        ),
    )
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
    assert segs[0]["words"][1][:4] == ["Logan", 1, 2, 0.2]
    assert segs[0]["words"][1][5] is None  # not corrected
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

    r = jcall(
        server,
        "/api/speakers/enroll",
        "POST",
        {"name": "Alice", "recording": 1, "segments": [2, 3], "threshold": 0.8},
    )
    assert r["labelled"] == 2  # re-labelled right away, no separate rescore needed
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


def test_rename(server):
    assert (
        jcall(server, "/api/recordings/1/rename", "POST", {"title": "Team sync"})["name"]
        == "Team sync"
    )
    (rec,) = jcall(server, "/api/recordings")
    assert (rec["name"], rec["file"], rec["title"]) == ("Team sync", "talk.m4a", "Team sync")
    assert jcall(server, "/api/search?q=there")[0]["name"] == "Team sync"
    jcall(server, "/api/recordings/1/rename", "POST", {"title": ""})
    assert jcall(server, "/api/recordings")[0]["name"] == "talk.m4a"


def wait_jobs(base):
    for _ in range(100):
        jobs = jcall(base, "/api/jobs")
        if all(j["status"] not in ("queued", "running") for j in jobs):
            return jobs
        time.sleep(0.05)
    raise AssertionError("jobs did not finish")


def test_import_transcribes_in_background(server, tmp_path):
    req = urllib.request.Request(
        server + "/api/import?name=..%2F..%2Fnew%20talk.m4a&threshold=0.8",
        data=b"new audio bytes",
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        job = json.loads(r.read())
    assert job["name"] == "new talk.m4a"  # directory part dropped
    (done,) = wait_jobs(server)
    assert done["status"] == "done" and done["recording"] == 2
    assert (tmp_path / "lib" / "new talk.m4a").read_bytes() == b"new audio bytes"
    segs = jcall(server, "/api/recordings/2/segments")
    assert [s["speaker"] for s in segs] == ["me", "other", "other"]  # labelled on import

    # the same bytes again: stored once, the extra copy removed
    with urllib.request.urlopen(
        urllib.request.Request(
            server + "/api/import?name=new%20talk.m4a", data=b"new audio bytes", method="POST"
        )
    ):
        pass
    dup = wait_jobs(server)[0]
    assert dup["status"] == "duplicate" and dup["recording"] == 2
    assert sorted(p.name for p in (tmp_path / "lib").iterdir()) == ["new talk.m4a"]


def test_import_rejects_non_audio(server):
    req = urllib.request.Request(server + "/api/import?name=notes.txt", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400


def test_live_session_over_http(server, tmp_path):
    sid = jcall(server, "/api/live/start", "POST", {"threshold": 0.3})["id"]
    audio = np.concatenate([tone(200, 2), tone(200, 2), tone(800, 2)])
    committed = []
    for i in range(0, len(audio), SR):
        req = urllib.request.Request(
            f"{server}/api/live/{sid}/audio",
            data=audio[i : i + SR].astype("<f4").tobytes(),
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            committed += json.loads(r.read())["committed"]
    # the fake numbers segments within the re-transcribed tail, so compare times
    assert [(c["n"], c["start"]) for c in committed] == [(1, 0.0), (2, 2.0)]
    stop = jcall(server, f"/api/live/{sid}/stop", "POST", {})
    assert [(c["n"], c["start"]) for c in stop["committed"]] == [(3, 4.0)]
    rec = [r for r in jcall(server, "/api/recordings") if r["id"] == stop["recording"]][0]
    assert rec["file"].startswith("live-") and rec["duration"] == 6.0
    segs = jcall(server, f"/api/recordings/{stop['recording']}/segments")
    assert [s["start"] for s in segs] == [0.0, 2.0, 4.0]
    assert all(s["speaker"] for s in segs)


def test_cli_ui_starts_server(tmp_path, monkeypatch, capsys):
    from recorder import web
    from recorder.cli import main

    started = {}

    def fake_serve_forever(self):
        started["app"] = self.RequestHandlerClass
        raise KeyboardInterrupt  # what Ctrl+C does

    monkeypatch.setattr(web.ThreadingHTTPServer, "serve_forever", fake_serve_forever)
    db = tmp_path / "t.db"
    main(
        [
            "--db",
            str(db),
            "ui",
            "--no-open",
            "--port",
            "0",
            "--model",
            "small",
            "--hotwords",
            "Logan",
            "--library",
            str(tmp_path / "lib"),
        ]
    )
    out = capsys.readouterr().out
    assert "recorder UI on http://127.0.0.1:" in out and str(tmp_path / "lib") in out
    assert started and db.exists()


def test_delete_recording(server, tmp_path):
    # a file outside the library: the recording goes, the file stays
    r = jcall(server, "/api/recordings/1", "DELETE")
    assert r == {"ok": True, "deleted_file": False, "path": str(tmp_path / "talk.m4a")}
    assert (tmp_path / "talk.m4a").exists()
    assert jcall(server, "/api/recordings") == [] and jcall(server, "/api/search?q=hello") == []

    # an imported file lives in the library: deleted with the recording
    req = urllib.request.Request(server + "/api/import?name=a.m4a", data=b"abc", method="POST")
    urllib.request.urlopen(req).read()
    (job,) = wait_jobs(server)
    r = jcall(server, f"/api/recordings/{job['recording']}", "DELETE")
    assert r["deleted_file"] and not (tmp_path / "lib" / "a.m4a").exists()

    with pytest.raises(urllib.error.HTTPError) as e:
        call(server, "/api/recordings/1", "DELETE")
    assert e.value.code == 404


def test_delete_many(server, tmp_path):
    req = urllib.request.Request(server + "/api/import?name=a.m4a", data=b"abc", method="POST")
    urllib.request.urlopen(req).read()
    (job,) = wait_jobs(server)
    r = jcall(server, "/api/recordings/delete", "POST", {"ids": [1, job["recording"], 99]})
    assert r == {"deleted": 2, "kept_files": [str(tmp_path / "talk.m4a")]}  # 99: not there
    assert jcall(server, "/api/recordings") == []
    assert (tmp_path / "talk.m4a").exists() and not (tmp_path / "lib" / "a.m4a").exists()


def test_unsure_word_candidates_and_correction(server, tmp_path):
    wid = jcall(server, "/api/recordings/1/segments")[0]["words"][1][4]
    FakeTranscriber.calls = 0
    info = jcall(server, f"/api/words/{wid}/candidates")
    assert jcall(server, f"/api/words/{wid}/candidates") == info
    assert FakeTranscriber.calls == 1  # worked out once, then read from the database
    assert (info["heard"], info["corrected"], info["probability"]) == ("logan", False, 0.2)
    words = [c["word"] for c in info["candidates"]]
    assert words[:2] == ["logging", "lagoon"] and len(words) <= 4 and "logan" not in words

    c = jcall(server, f"/api/words/{wid}/correct", "POST", {"word": "Logging"})
    assert (c["heard"], c["chosen"], c["word"]) == ("logan", "logging", "Logging")
    assert c["compare"]["target_ipa"] and c["compare"]["ops"]
    seg = jcall(server, "/api/recordings/1/segments")[0]
    assert seg["text"] == "hello Logging" and seg["words"][1][5] == "Logan"
    assert jcall(server, f"/api/words/{wid}/candidates")["corrected"]

    (row,) = jcall(server, "/api/corrections")
    assert (row["chosen"], row["heard"], row["recording"]) == ("logging", "logan", "talk.m4a")
    md = call(server, "/api/corrections.md")[2].decode()
    assert "| logging | logan |" in md and "hello Logging" in md

    jcall(server, f"/api/words/{wid}/correct", "POST", {"word": "logan"})  # it was right
    (row,) = jcall(server, "/api/corrections")  # kept, to compare pronunciations
    assert (row["chosen"], row["heard"]) == ("logan", "logan")
    md = call(server, "/api/corrections.md")[2].decode()
    assert md.startswith("# Pronunciation notebook") and "| logan | (right, but" in md
    md = call(server, "/api/corrections.md?lang=zh")[2].decode()
    assert md.startswith("# 纠错本") and "| logan | （识别对了，但没把握） |" in md


def test_pronunciation_proxy(server, tmp_path):
    status, headers, body = call(server, "/api/pronounce?word=Logan")
    assert status == 200 and body == b"MP3" and headers["Content-Type"] == "audio/mpeg"
    assert (tmp_path / "pronunciations" / "logan.mp3").read_bytes() == b"MP3"
    for word, code in [("nothing", 404), ("..%2F1", 400)]:
        with pytest.raises(urllib.error.HTTPError) as e:
            call(server, f"/api/pronounce?word={word}")
        assert e.value.code == code


def test_unsure_words_are_analysed_in_one_go(tmp_path, monkeypatch):
    audio = np.zeros(6 * SR, np.float32)
    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: audio)
    (tmp_path / "a.wav").write_bytes(b"x")
    words = [Word(0, 1, "I", 0.9), Word(1, 2, "sink", 0.3), Word(2, 3, "so", 0.5)]
    t = Transcript("en", 6, [Segment(0, 3, "I sink so", words=words), Segment(3, 6, "fine")])
    db = tmp_path / "t.db"
    with Store(db) as store:
        store.save(tmp_path / "a.wav", "h", t)

    class T(FakeTranscriber):
        def alternatives(self, audio, spans, language=None):
            T.calls += 1
            assert len(audio) == 3 * SR and spans == [(1, 2), (2, 3)]  # both words, one pass
            return [["think", "think"], ["sew"]]

    app = App(db, transcriber=T(), library=tmp_path / "lib")
    T.calls = 0
    (job,) = app.suggest_pending()
    assert (job["kind"], job["recording"]) == ("suggest", 1)
    for _ in range(100):
        if job["status"] == "done":
            break
        time.sleep(0.02)
    assert job["status"] == "done" and job["progress"] == [2, 2] and T.calls == 1
    with Store(db) as store:
        ids = [w.id for w in store.segments(1, words=True)[0].words]
        assert store.candidates(ids[0]) is None  # sure word: not analysed
        assert store.candidates(ids[1])[0] == "think" and store.candidates(ids[2])[0] == "sew"
        assert store.pending_unsure() == {}
    assert app.suggest_pending() == []  # nothing left to do


def test_word_clip_is_cut_exactly(server):
    import io
    import wave

    wid = jcall(server, "/api/recordings/1/segments")[0]["words"][1][4]  # "Logan", 1-2 s
    status, headers, body = call(server, f"/api/words/{wid}/clip.wav")
    assert status == 200 and headers["Content-Type"] == "audio/wav"
    with wave.open(io.BytesIO(body)) as f:
        assert f.getframerate() == SR and f.getnframes() == int(1.1 * SR)  # 1 s + 0.05 s each side


def test_synthesised_voices(tmp_path, monkeypatch):
    import io
    import wave

    from recorder import tts

    class FakeVoices:
        def __init__(self, folder):
            assert folder == tmp_path / "models" / "kokoro"

        def say(self, word, voice):
            # 0.5 s silence, 0.3 s "speech", 0.5 s silence
            s = np.zeros(int(1.3 * 24000), np.float32)
            s[12000:19200] = 0.5 if voice == "female" else -0.5
            return s, 24000

    monkeypatch.setattr(tts, "KokoroVoices", FakeVoices)
    app = App(tmp_path / "t.db", library=tmp_path / "lib")
    data, ctype = app.pronunciation("Think,", "female")
    assert ctype == "audio/wav"
    with wave.open(io.BytesIO(data)) as f:
        assert (
            f.getframerate() == 24000 and f.getnframes() == 7200 + 2 * 1200
        )  # trimmed, 50 ms pads
    assert (tmp_path / "pronunciations" / "female" / "think.wav").read_bytes() == data
    assert app.pronunciation("think", "male")[0] != data
    with pytest.raises(Exception) as e:
        app.pronunciation("think", "robot")
    assert e.value.status == 400


def test_set_start_time(server):
    assert jcall(server, "/api/recordings")[0]["started_at"] is None
    r = jcall(server, "/api/recordings/1/started", "POST", {"started_at": "2026-10-16T09:15"})
    assert r["started_at"] == "2026-10-16T09:15:00"
    assert jcall(server, "/api/recordings")[0]["started_at"] == "2026-10-16T09:15:00"
    assert jcall(server, "/api/search?q=there")[0]["started_at"] == "2026-10-16T09:15:00"
    with pytest.raises(urllib.error.HTTPError) as e:
        call(server, "/api/recordings/1/started", "POST", {"started_at": "yesterday"})
    assert e.value.code == 400
    jcall(server, "/api/recordings/1/started", "POST", {"started_at": ""})
    assert jcall(server, "/api/recordings")[0]["started_at"] is None
