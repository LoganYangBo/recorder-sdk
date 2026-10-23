import asyncio
from datetime import datetime

import numpy as np
import pytest

from recorder import ble
from recorder.ble import BleRecorder, RecorderError, download_new, parse_status


class FakeRecorder:
    """Speaks the firmware's Bluetooth protocol from a dict of files."""

    def __init__(self, files, chunk=240, lose=()):
        self.files = dict(files)
        self.chunk = chunk
        self.lose = set(lose)  # offsets of DATA packets to drop (once each)
        self.cb = {}
        self.sent = []
        self.clock = None
        self.is_connected = False

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def start_notify(self, uuid, cb):
        self.cb[uuid] = cb

    def _ctrl(self, line):
        self.cb[ble.CTRL](None, bytearray(line.encode()))

    async def write_gatt_char(self, uuid, data, response=None):
        assert uuid == ble.CTRL
        line = data.decode()
        self.sent.append(line)
        if line == "?":
            self._ctrl("RECORDER 3 20261020T093000 LISTEN - dropped=0 bat=3.91 pct=65")
        elif line.startswith("T"):
            self.clock = int(line[1:])
            self._ctrl("OK 20261020T093000")
        elif line == "X":
            self._ctrl("STOPPED")
        elif line == "L":
            for path, data in self.files.items():
                self._ctrl(f"F {path} {len(data)}")
            self._ctrl("END")
        elif line.startswith("G"):
            off, path = line[1:].split(" ", 1)
            if path not in self.files:
                return self._ctrl("ERR no such file")
            data, off = self.files[path], int(off)
            while off < len(data):
                piece = data[off : off + self.chunk]
                if off in self.lose:
                    self.lose.discard(off)
                else:
                    self.cb[ble.DATA](None, bytearray(off.to_bytes(4, "little") + piece))
                off += len(piece)
            self._ctrl(f"DONE {len(data)}")
        elif line.startswith("D"):
            self._ctrl("OK" if self.files.pop(line[1:], None) is not None else "ERR not removed")
        elif line in ("S1", "S0"):
            self._ctrl("OK")


def run(coro):
    return asyncio.run(coro)


async def _with(fake, work):
    """Run `work(recorder)` against the fake."""
    rec = BleRecorder(fake, timeout=1)
    await rec.start()
    return await work(rec)


def test_parse_status():
    s = parse_status("RECORDER 3 20261020T093000 REC REC/20261020/093000.WAV dropped=0 pct=65")
    assert s["clock"] == "20261020T093000" and s["state"] == "REC"
    assert s["file"] == "REC/20261020/093000.WAV" and s["pct"] == "65"


def test_status_and_clock():
    fake = FakeRecorder({})

    async def go(rec):
        return await rec.status(), await rec.set_clock(datetime(2026, 10, 20, 9, 30))

    status, stamp = run(_with(fake, go))
    assert status["pct"] == "65" and status["bat"] == "3.91"
    assert stamp == "20261020T093000"
    assert fake.sent[-1] == "T1792488600"


def test_list_and_fetch(tmp_path):
    files = {"REC/20261020/093000.WAV": bytes(range(256)) * 9, "REC/NOTIME/00000001.WAV": b"x"}
    fake = FakeRecorder(files)

    async def go(rec):
        listed = await rec.list_files()
        dest = tmp_path / "a.wav"
        await rec.fetch("REC/20261020/093000.WAV", len(files["REC/20261020/093000.WAV"]), dest)
        return listed, dest.read_bytes()

    listed, got = run(_with(fake, go))
    assert listed == [(p, len(d)) for p, d in files.items()]
    assert got == files["REC/20261020/093000.WAV"]


def test_fetch_recovers_a_lost_packet(tmp_path):
    data = bytes(range(256)) * 10
    fake = FakeRecorder({"REC/A/B.WAV": data}, lose={480})

    async def go(rec):
        await rec.fetch("REC/A/B.WAV", len(data), tmp_path / "b.wav")

    run(_with(fake, go))
    assert (tmp_path / "b.wav").read_bytes() == data
    assert [s for s in fake.sent if s.startswith("G")] == ["G0 REC/A/B.WAV", "G480 REC/A/B.WAV"]


def test_fetch_resumes_a_partial_download(tmp_path):
    data = b"abcdefghij" * 100
    (tmp_path / "c.wav.part").write_bytes(data[:300])
    fake = FakeRecorder({"REC/A/C.WAV": data})
    run(_with(fake, lambda rec: rec.fetch("REC/A/C.WAV", len(data), tmp_path / "c.wav")))
    assert (tmp_path / "c.wav").read_bytes() == data
    assert not (tmp_path / "c.wav.part").exists()
    assert "G300 REC/A/C.WAV" in fake.sent


def test_fetch_missing_file_raises(tmp_path):
    fake = FakeRecorder({})
    with pytest.raises(RecorderError):
        run(_with(fake, lambda rec: rec.fetch("REC/A/X.WAV", 5, tmp_path / "x.wav")))


def test_download_new_skips_files_already_there(tmp_path):
    files = {"REC/20261020/093000.WAV": b"1" * 500, "REC/20261020/094500.WAV": b"2" * 700}
    (tmp_path / "REC/20261020").mkdir(parents=True)
    (tmp_path / "REC/20261020/093000.WAV").write_bytes(files["REC/20261020/093000.WAV"])
    fake = FakeRecorder(files)
    got = run(_with(fake, lambda rec: download_new(rec, tmp_path)))
    assert [p for p, _ in got] == list(files)
    assert (tmp_path / "REC/20261020/094500.WAV").read_bytes() == files["REC/20261020/094500.WAV"]
    assert [s for s in fake.sent if s.startswith("G")] == ["G0 REC/20261020/094500.WAV"]


def test_delete():
    fake = FakeRecorder({"REC/A/B.WAV": b"x"})
    run(_with(fake, lambda rec: rec.delete("REC/A/B.WAV")))
    assert fake.files == {}
    with pytest.raises(RecorderError):
        run(_with(fake, lambda rec: rec.delete("REC/A/B.WAV")))


def test_live_decodes_blocks_and_counts_losses():
    fake = FakeRecorder({})
    got = []

    async def go(rec):
        await rec.live(got.append)
        for seq in (0, 1, 3):  # 2 went missing
            block = bytes([0x10, 0x00, 0, seq]) + bytes(8)
            fake.cb[ble.LIVE](None, bytearray(block))
        await rec.live(None)
        return rec.live_lost

    lost = run(_with(fake, go))
    assert lost == 1
    assert len(got) == 3 and all(len(b) == 17 and b.dtype == np.int16 for b in got)
    assert fake.sent[-2:] == ["S1", "S0"]


def test_link_runs_calls_on_its_own_loop():
    fake = FakeRecorder({"REC/A/B.WAV": b"xyz"})

    async def finder():
        return "device"

    link = ble.BleLink(finder=finder, make_client=lambda dev: fake)
    with pytest.raises(RecorderError):
        link.call(lambda rec: rec.status())  # not connected yet
    assert link.connect() == "20261020T093000"
    assert link.connected and fake.sent[0].startswith("T")
    assert link.call(lambda rec: rec.list_files()) == [("REC/A/B.WAV", 3)]
    link.disconnect()
    assert not link.connected and not fake.is_connected


def test_link_without_a_recorder_in_range():
    async def finder():
        return None

    with pytest.raises(RecorderError, match="no recorder"):
        ble.BleLink(finder=finder).connect()


def test_cli_sync_over_bluetooth_transcribes_as_files_arrive(tmp_path, monkeypatch, capsys):
    pytest.importorskip("av")
    from test_adpcm import ima_wav
    from test_sync import FakeTranscriber

    from recorder import Store, cli

    blocks = [bytes([0, 0, 10, 0]) + bytes(range(i, i + 252)) for i in range(3)]
    fake = FakeRecorder(
        {
            "REC/20261020/093000.WAV": ima_wav(blocks),
            "REC/20261020/094000.WAV": ima_wav(blocks[::-1]),
        }
    )
    order = []

    class Tracking(FakeTranscriber):
        def transcribe(self, path):
            order.append(("transcribe", len(fake.sent)))
            return super().transcribe(path)

    monkeypatch.setattr(cli, "_ble", lambda args, work: run(_with(fake, work)))
    monkeypatch.setattr(cli, "_transcriber", lambda args: Tracking())
    db = str(tmp_path / "t.db")
    cli.main(["--db", db, "sync", "--ble", "--no-speakers", "--no-suggest", "--delete"])
    out = capsys.readouterr().out
    assert "bluetooth: 2 new" in out and "deleted 2 synced recording(s)" in out
    assert len(order) == 2 and fake.files == {}
    with Store(db) as store:
        assert [r.started_at for r in store.recordings()] == [
            "2026-10-20T09:30:00",
            "2026-10-20T09:40:00",
        ]
