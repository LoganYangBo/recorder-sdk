import json
import wave

import numpy as np
import pytest

from recorder import FolderSource, Pipeline, Segment, Store, Syncer, Transcript, find_devices
from recorder.sync import parse_started_at


@pytest.mark.parametrize(
    "name, expected",
    [
        ("20261016T091500.wav", "2026-10-16T09:15:00"),
        ("REC_20261016_091500.m4a", "2026-10-16T09:15:00"),
        ("2026-10-16 09.15.00.m4a", "2026-10-16T09:15:00"),
        ("Recording 2026-10-16T09:15:00.wav", "2026-10-16T09:15:00"),
        ("20261399T999999.wav", None),  # not a real date
        ("demo-1.m4a", None),
        ("123420261016T0915001.wav", None),  # digits run on: not a timestamp
    ],
)
def test_parse_started_at(name, expected):
    assert parse_started_at(name) == expected


def test_parse_started_at_converts_utc_to_local(monkeypatch):
    import os
    import time

    monkeypatch.setenv("TZ", "Asia/Shanghai")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        if os.name != "nt":
            assert parse_started_at("20261016T011500Z.wav") == "2026-10-16T09:15:00"
    finally:
        monkeypatch.undo()
        if hasattr(time, "tzset"):
            time.tzset()


def write_wav(path, seconds=1.0, freq=220):
    t = np.arange(int(16000 * seconds)) / 16000
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2").tobytes())


class FakeTranscriber:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def transcribe(self, path):
        self.calls.append(path)
        if any(f in str(path) for f in self.fail):
            raise RuntimeError("corrupt file")
        return Transcript("en", 1.0, [Segment(0, 1, f"text of {path.name}")])


def make_device(root, names):
    (root / "REC").mkdir(parents=True)
    (root / "recorder-device.json").write_text(json.dumps({"id": "xiao-1"}))
    for i, n in enumerate(names):
        write_wav(root / "REC" / n, freq=200 + 50 * i)
    return root


def test_find_devices(tmp_path):
    dev = make_device(tmp_path / "E", ["20261016T091500.wav"])
    (tmp_path / "USB").mkdir()
    (d,) = find_devices([dev, tmp_path / "USB", tmp_path / "missing"])
    assert d.id == "xiao-1" and d.root == dev / "REC"
    (f,) = d.list_files()
    assert f.started_at == "2026-10-16T09:15:00"


def test_sync_copies_compresses_and_dedupes(tmp_path):
    pytest.importorskip("av")
    dev = make_device(tmp_path / "E", ["20261016T091500.wav", "20261016T101500.wav"])
    (d,) = find_devices([dev])
    fake = FakeTranscriber()
    with Store(tmp_path / "t.db") as store:
        syncer = Syncer(store, Pipeline(fake, store), tmp_path / "lib")
        assert [r.status for r in syncer.sync(d)] == ["new", "new"]
        rec = store.recordings()[0]
        assert rec.started_at == "2026-10-16T09:15:00"
        copy = tmp_path / "lib" / "2026" / "10" / "16" / "20261016T091500.opus"
        assert (
            rec.path == str(copy)
            and copy.stat().st_size < (dev / "REC" / "20261016T091500.wav").stat().st_size / 4
        )
        assert not list((tmp_path / "lib").rglob("*.part"))
        # the device still has its files; syncing again does nothing new
        assert [r.status for r in syncer.sync(d)] == ["known", "known"]
        assert len(fake.calls) == 2 and [f["status"] for f in store.files()] == ["done", "done"]


def test_failures_are_kept_and_retried(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_wav(src / "20261016T091500.wav")
    write_wav(src / "20261016T101500.wav", freq=300)
    fake = FakeTranscriber(fail=["101500"])
    with Store(tmp_path / "t.db") as store:
        syncer = Syncer(
            store, Pipeline(fake, store), tmp_path / "lib", compress=False, max_attempts=2
        )
        assert [r.status for r in syncer.sync(FolderSource(src))] == ["new", "failed"]
        (failed,) = store.files("failed")
        assert failed["error"] == "RuntimeError: corrupt file" and failed["attempts"] == 1
        assert [r.status for r in syncer.sync(FolderSource(src))] == ["known", "failed"]
        assert [r.status for r in syncer.sync(FolderSource(src))] == ["known", "gave-up"]
        assert len(fake.calls) == 3  # the copy is reused, not made again
        fake.fail.clear()
        assert [r.status for r in syncer.sync(FolderSource(src), retry=True)] == ["known", "new"]
        assert store.files("failed") == [] and len(store.recordings()) == 2


def test_delete_source_only_once_synced(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_wav(src / "20261016T091500.wav")
    write_wav(src / "20261016T101500.wav", freq=300)
    fake = FakeTranscriber(fail=["101500"])
    with Store(tmp_path / "t.db") as store:
        syncer = Syncer(
            store, Pipeline(fake, store), tmp_path / "lib", compress=False, delete_source=True
        )
        syncer.sync(FolderSource(src))
    assert [p.name for p in src.iterdir()] == ["20261016T101500.wav"]  # the failed one stays


def test_process_source_keeps_going_after_a_failure(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.wav").write_bytes(b"a")
    (src / "b.wav").write_bytes(b"b")
    with Store(tmp_path / "t.db") as store:
        results = Pipeline(FakeTranscriber(fail=["a.wav"]), store).process_source(FolderSource(src))
    assert results[0].error == "RuntimeError: corrupt file" and results[0].recording_id is None
    assert results[1].recording_id == 1 and results[1].error is None


def test_cli_sync_folder(tmp_path, monkeypatch, capsys):
    from recorder import cli

    src = tmp_path / "src"
    src.mkdir()
    write_wav(src / "20261016T091500.wav")
    monkeypatch.setattr(cli, "_transcriber", lambda args: FakeTranscriber())
    db = str(tmp_path / "t.db")
    cli.main(["--db", db, "sync", str(src), "--keep-wav", "--no-speakers", "--no-suggest"])
    assert "1 new" in capsys.readouterr().out
    cli.main(["--db", db, "status"])
    assert capsys.readouterr().out.strip() == "1 done"
    cli.main(["--db", db, "list"])
    assert "2026-10-16 09:15" in capsys.readouterr().out


def test_cli_dates_backfills(tmp_path, capsys):
    from recorder import cli

    (tmp_path / "20261016T091500.wav").write_bytes(b"x")
    (tmp_path / "demo.wav").write_bytes(b"y")
    with Store(tmp_path / "t.db") as store:
        store.save(tmp_path / "20261016T091500.wav", "a", Transcript("en", 1))
        store.save(tmp_path / "demo.wav", "b", Transcript("en", 1))
    cli.main(["--db", str(tmp_path / "t.db"), "dates"])
    with Store(tmp_path / "t.db") as store:
        assert [r.started_at for r in store.recordings()] == ["2026-10-16T09:15:00", None]
