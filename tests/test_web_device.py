"""The web UI's Bluetooth page, against a fake recorder."""

import time

import numpy as np
import pytest
from test_adpcm import ima_wav
from test_ble import FakeRecorder
from test_web import SR, BandEmbedder, FakeTranscriber, jcall, tone

from recorder import Store, ble
from recorder import speaker as speaker_mod
from recorder.web import App, make_handler


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer

    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: tone(200, 6))
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(40):
        b = bytearray(rng.integers(0, 16, 256, dtype=np.uint8).tobytes())
        b[2], b[3] = 10, 0
        blocks.append(bytes(b))
    card = {"REC/20261020/093000.WAV": ima_wav(blocks)}
    fake = FakeRecorder(card)

    async def finder():
        return "device"

    db = tmp_path / "t.db"
    with Store(db) as store:
        store.save_voiceprint("me", BandEmbedder.name, np.array([1, 0], np.float32).tobytes())
    app = App(
        db,
        embedder=BandEmbedder(),
        transcriber=FakeTranscriber(),
        library=tmp_path / "lib",
        link=ble.BleLink(finder=finder, make_client=lambda dev: fake),
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", fake, tmp_path
    srv.shutdown()
    srv.server_close()


def wait_job(base, kind):
    for _ in range(100):
        jobs = [j for j in jcall(base, "/api/jobs") if j["kind"] == kind]
        if jobs and jobs[0]["status"] not in ("queued", "running"):
            return jobs[0]
        time.sleep(0.05)
    raise AssertionError("job didn't finish")


def test_connect_status_and_disconnect(setup):
    base, fake, _ = setup
    assert jcall(base, "/api/device")["connected"] is False
    d = jcall(base, "/api/device/connect", "POST", {})
    assert d["connected"] and d["clock"] == "20261020T093000"
    assert d["status"]["pct"] == "65" and d["files"]["count"] == 1
    assert jcall(base, "/api/device")["status"]["state"] == "LISTEN"
    assert jcall(base, "/api/device/disconnect", "POST", {})["connected"] is False


def test_sync_downloads_transcribes_and_deletes(setup):
    av = pytest.importorskip("av")  # noqa: F841 (compresses to Opus)
    base, fake, tmp = setup
    jcall(base, "/api/device/connect", "POST", {})
    jcall(base, "/api/device/sync", "POST", {"delete": True})
    job = wait_job(base, "device")
    assert job["status"] == "done", job
    assert job["new"] == 1 and job["progress"] == [1, 1]
    rec = [r for r in jcall(base, "/api/recordings") if r["id"] == job["recording"]][0]
    assert rec["started_at"] == "2026-10-20T09:30:00"
    assert fake.files == {}  # deleted from the card
    assert not (tmp / "inbox/REC/20261020/093000.WAV").exists()
    assert jcall(base, "/api/device")["files"]["count"] == 0


def test_live_transcription(setup):
    base, fake, _ = setup
    jcall(base, "/api/device/connect", "POST", {})
    jcall(base, "/api/device/live/start", "POST", {})
    assert "S1" in fake.sent and jcall(base, "/api/device")["live"]
    pcm = (tone(200, 5) * 20000).astype(np.int16)
    block = 473
    for i in range(0, len(pcm) - block, block):  # ADPCM-free: blocks of raw first samples
        x = pcm[i : i + block]
        data = x[0].astype("<i2").tobytes() + bytes([0, i // block % 256]) + bytes(236)
        fake.cb[ble.LIVE](None, bytearray(data))
    for _ in range(100):
        r = jcall(base, "/api/device/live?from=0")
        if r["committed"]:
            break
        time.sleep(0.05)
    assert r["on"] and r["committed"][0]["text"] == "live 0"
    assert r["duration"] == pytest.approx(len(pcm) // block * block / SR, abs=0.1)
    stop = jcall(base, "/api/device/live/stop", "POST", {})
    assert stop["recording"] is None  # the card has the same audio
    assert jcall(base, "/api/recordings") == []
    assert "S0" in fake.sent and not jcall(base, "/api/device")["live"]
