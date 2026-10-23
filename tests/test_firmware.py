"""The firmware itself, compiled for this computer (firmware/sim): its microphone plays
generated sound, its card is a folder, and the SDK's Bluetooth code talks to it."""

import asyncio
import shutil
import struct
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from recorder import ble
from recorder.adpcm import decode_block
from recorder.ble import BleRecorder, download_new, parse_status
from recorder.clock import local_epoch

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
CXX = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
pytestmark = pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")
RATE = 16000
CHUNK = RATE * 600  # samples in a full file


@pytest.fixture(scope="session")
def sim_exe(tmp_path_factory):
    exe = tmp_path_factory.mktemp("sim") / "sim.exe"
    cmd = [CXX, "-std=c++17", "-O2", "-DUSE_SCREEN=0", f"-I{FIRMWARE / 'sim'}"]
    cmd += ["-o", str(exe), str(FIRMWARE / "sim" / "sim.cpp")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return exe


class Sim:
    """The firmware running in a subprocess, driven by sim.cpp's commands."""

    def __init__(self, exe, card: Path, *args: str):
        self.card = card
        self.p = subprocess.Popen(
            [str(exe), "--card", str(card), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.notes: list[tuple[str, bytes]] = []
        self.serial: list[str] = []
        self.stats: dict[str, int] = {}

    def cmd(self, line: str) -> list[tuple[str, bytes]]:
        """Run a command; returns the notifications sent meanwhile."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        sent = []
        while True:
            out = self.p.stdout.readline()
            assert out, f"the simulator stopped during {line!r}"
            kind, _, rest = out.rstrip("\n").partition(" ")
            if kind == ".":
                self.notes += sent
                return sent
            if kind == "N":
                chr_, _, data = rest.partition(" ")
                sent.append((chr_, bytes.fromhex(data)))
            elif kind == "S":
                self.serial.append(rest)
            elif kind == "STATS":
                self.stats = {k: int(v) for k, v in (f.split("=") for f in rest.split())}

    def play(self, tmp: Path, *pieces: np.ndarray) -> None:
        path = tmp / f"audio{len(list(tmp.glob('audio*')))}.raw"
        np.concatenate(pieces).astype("<i2").tofile(path)
        self.cmd(f"audio {path}")

    def run(self, seconds: float) -> None:
        self.cmd(f"run {int(seconds * 1000)}")

    def status(self) -> dict:
        self.cmd("serial ?")
        return parse_status(self.serial[-1])

    def get_stats(self) -> dict:
        self.cmd("stats")
        return self.stats

    def close(self):
        self.p.stdin.close()
        self.p.wait(10)


class SimClient:
    """A bleak client whose recorder is the simulator."""

    UUIDS = {"CTRL": ble.CTRL, "DATA": ble.DATA, "LIVE": ble.LIVE}

    def __init__(self, sim: Sim, limit_ms: int | None = None):
        self.sim = sim
        self.cb = {}
        self.limit_ms = limit_ms  # cut a command short (as if the link dropped)

    async def start_notify(self, uuid, cb):
        self.cb[uuid] = cb

    async def write_gatt_char(self, uuid, data, response=None):
        assert uuid == ble.CTRL
        line = data.decode()
        cmd = f"writefor {self.limit_ms} {line}" if self.limit_ms else f"write {line}"
        for chr_, payload in self.sim.cmd(cmd):
            self.cb[self.UUIDS[chr_]](None, bytearray(payload))


@pytest.fixture
def sim(sim_exe, tmp_path):
    sims = []

    def start(*args: str) -> Sim:
        s = Sim(sim_exe, tmp_path / f"card{len(sims)}", *args)
        sims.append(s)
        return s

    yield start
    for s in sims:
        s.close()


def quiet(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(-2, 3, int(seconds * RATE))


def talk(seconds: float, amp: float = 3000) -> np.ndarray:
    """Something loud and voice-like: a tone with a syllable-rate wobble."""
    t = np.arange(int(seconds * RATE)) / RATE
    return amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))


def read_wav(data: bytes) -> tuple[dict, np.ndarray]:
    """The firmware's IMA ADPCM WAV: header fields and decoded samples."""
    assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt "
    riff = struct.unpack_from("<I", data, 4)[0]
    fmt, ch, rate, _, block, bits, _, spb = struct.unpack_from("<HHIIHHHH", data, 20)
    assert data[40:44] == b"fact" and data[52:56] == b"data"
    fact, size = struct.unpack_from("<I", data, 48)[0], struct.unpack_from("<I", data, 56)[0]
    body = data[60 : 60 + size]
    pcm = np.concatenate([decode_block(body[i : i + block]) for i in range(0, size, block)])
    head = dict(riff=riff, format=fmt, channels=ch, rate=rate, block=block, bits=bits)
    head.update(samples_per_block=spb, fact=fact, size=size)
    return head, pcm


def connect(sim: Sim, work, limit_ms=None):
    """Connect the simulator and run `await work(BleRecorder)`."""

    async def go():
        rec = BleRecorder(SimClient(sim, limit_ms), timeout=1)
        await rec.start()
        return await work(rec)

    sim.cmd("connect")
    return asyncio.run(go())


def wavs(card: Path) -> list[Path]:
    return sorted((card / "REC").rglob("*.WAV"))


def test_records_a_stretch_of_sound_and_syncs_it(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(2), talk(4), quiet(1))
    s.run(20)
    (f,) = wavs(s.card)
    assert f.relative_to(s.card).as_posix() == "REC/NOTIME/00000001.WAV"
    head, pcm = read_wav(f.read_bytes())
    assert head["riff"] == f.stat().st_size - 8 and head["size"] == f.stat().st_size - 60
    assert head["format"] == 0x11 and head["rate"] == RATE and head["block"] == 256
    assert head["fact"] == head["size"] // 256 * 505 == len(pcm)
    # a second before the sound, the sound, then 10 s of quiet
    assert 14.5 < len(pcm) / RATE < 15.6
    loud = np.flatnonzero(np.abs(pcm) > 1000)
    assert abs(loud[0] / RATE - 1.0) < 0.1 and abs(loud[-1] / RATE - 5.0) < 0.1

    inbox = tmp_path / "inbox"
    got = connect(s, lambda rec: download_new(rec, inbox))
    assert [p for p, _ in got] == ["REC/NOTIME/00000001.WAV"]
    assert (inbox / "REC/NOTIME/00000001.WAV").read_bytes() == f.read_bytes()
    assert (s.card / "recorder-device.json").read_text() == '{"id": "recorder-c05e"}\n'


def test_card_writes_end_on_sector_boundaries(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(95), quiet(1))
    s.run(110)
    st = s.get_stats()
    # only the writes before a header save (every 30 s, and at the end) may stop mid-sector
    assert st["writes"] > 40 and st["unaligned"] <= 5
    assert st["dropped"] == 0 and st["barks"] == 0
    assert st["gain"] == 0x28 and st["dcdc"] == 1


def test_clock_names_files_and_long_sound_is_split(sim, tmp_path):
    s = sim()
    connect(s, lambda rec: rec.set_clock(datetime(2026, 10, 20, 9, 30, 0)))
    s.cmd("disconnect")
    s.play(tmp_path, quiet(2), talk(1250), quiet(1))
    s.run(1270)
    files = [f.relative_to(s.card).as_posix() for f in wavs(s.card)]
    # the first starts with the second kept from before the sound (at 09:30:02)
    assert files == [f"REC/20261020/09{m}01.WAV" for m in ("30", "40", "50")]
    lens = [read_wav(f.read_bytes())[0]["fact"] for f in wavs(s.card)]
    assert lens[0] >= CHUNK and lens[1] >= CHUNK and lens[0] - CHUNK < 505
    assert s.status()["clock"].startswith("20261020T0951")


def test_clock_set_while_recording_renames_the_rest(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(10), quiet(1))
    s.run(4)
    connect(s, lambda rec: rec.set_clock(datetime(2026, 10, 20, 9, 30, 0)))
    s.run(20)
    files = [f.relative_to(s.card).as_posix() for f in wavs(s.card)]
    assert files == ["REC/20261020/093000.WAV", "REC/NOTIME/00000001.WAV"]


def test_a_knock_is_not_kept(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(2), talk(0.3), quiet(2))
    s.run(20)
    assert wavs(s.card) == []


def test_transfer_while_recording_loses_no_audio(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(30), quiet(12), talk(60), quiet(12))
    s.run(50)  # the first file is done, the second is being recorded
    first = wavs(s.card)[0]

    async def work(rec):
        files = await rec.list_files()
        await rec.fetch(*files[0], tmp_path / "got.wav")
        return files, await rec.status()

    files, status = connect(s, work)
    assert [p for p, _ in files] == [first.relative_to(s.card).as_posix()]  # not the open one
    assert (tmp_path / "got.wav").read_bytes() == first.read_bytes()
    assert status["state"] == "REC" and status["dropped"] == "0" and status["txq"] == "16"
    s.run(70)
    assert s.get_stats()["dropped"] == 0
    head, pcm = read_wav(wavs(s.card)[1].read_bytes())
    assert 70.5 < len(pcm) / RATE < 71.6  # 1 s + 60 s + 10 s, none of it lost


def test_download_resumes_after_the_link_drops(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(20), quiet(1))
    s.run(35)
    (f,) = wavs(s.card)
    path = f.relative_to(s.card).as_posix()
    dest = tmp_path / "got.wav"

    async def cut(rec):
        with pytest.raises(ble.RecorderError):
            await rec.fetch(path, f.stat().st_size, dest)

    connect(s, cut, limit_ms=300)  # the link "drops" 0.3 s into the download
    s.cmd("disconnect")
    part = dest.with_name("got.wav.part")
    assert 0 < part.stat().st_size < f.stat().st_size
    connect(s, lambda rec: rec.fetch(path, f.stat().st_size, dest))
    assert dest.read_bytes() == f.read_bytes()


def test_delete_takes_the_emptied_folder(sim, tmp_path):
    s = sim()
    connect(s, lambda rec: rec.set_clock(datetime(2026, 10, 20, 9, 30, 0)))
    s.cmd("disconnect")
    s.play(tmp_path, quiet(1), talk(3), quiet(1))
    s.run(16)
    (f,) = wavs(s.card)

    async def work(rec):
        await rec.delete(f.relative_to(s.card).as_posix())
        return await rec.list_files()

    assert connect(s, work) == []
    assert not (s.card / "REC" / "20261020").exists() and (s.card / "REC").is_dir()


def test_live_audio(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(5))
    s.run(1)
    blocks = []

    async def work(rec):
        await rec.live(blocks.append)
        for chr_, data in s.cmd("run 3000"):
            if chr_ == "LIVE":
                rec._live(None, bytearray(data))
        return rec.live_lost

    assert connect(s, work) == 0
    pcm = np.concatenate(blocks)
    assert 2.8 < len(pcm) / RATE < 3.1 and np.abs(pcm).max() > 1000


def test_small_softdevice_ram_falls_back_to_the_usual_queue(sim):
    s = sim("--queue-limit", "8")
    assert s.status()["txq"] == "3"


@pytest.mark.parametrize(
    "reset, kept",
    [("watchdog", True), ("button", True), ("power", False), ("button,soft", False)],
)
def test_clock_survives_a_reset(sim, reset, kept):
    t = local_epoch(datetime(2026, 10, 20, 9, 30, 0))
    status = sim("--reset", reset, "--kept", str(t)).status()
    assert status["clock"] == ("20261020T093001" if kept else "unset")


def test_card_failure_is_retried(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(10), quiet(80))
    s.run(4)
    s.cmd("card broken")
    s.run(2)
    assert s.status()["state"] == "FAILED"
    s.cmd("card ok")
    s.run(65)
    assert s.status()["state"] == "LISTEN"
    assert s.get_stats()["barks"] == 0


def test_button_pauses_and_hold_sleeps(sim, tmp_path):
    s = sim()
    s.play(tmp_path, quiet(1), talk(5), quiet(30))
    s.run(3)
    s.cmd("press 200")
    assert s.status()["state"] == "PAUSED"
    assert len(wavs(s.card)) == 1  # what was recorded so far is kept
    s.cmd("press 200")
    assert s.status()["state"] == "LISTEN"
    s.cmd("press 2500")
    assert s.status()["state"] == "PAUSED"  # asleep
    s.cmd("press 200")
    assert s.status()["state"] == "LISTEN"
