"""Talk to the recorder over Bluetooth LE: clock, file sync and live audio.

The firmware (firmware/recorder) offers one service with three characteristics:
CTRL takes a command line and notifies the replies, one line each; DATA
notifies file contents as packets of a 4-byte little-endian offset plus bytes;
LIVE notifies live audio, one IMA ADPCM block per packet (see adpcm.py).

    ?                 -> RECORDER 3 <stamp|unset> <state> <file|-> ... bat=<V> pct=<%>
    T<seconds>        -> OK <stamp>          (local wall-clock time, as if UTC)
    L                 -> F <path> <size> ... END   (finished files only)
    G<offset> <path>  -> DATA packets, then DONE <size>, or ERR ...
    X                 -> STOPPED             (stops a listing or download)
    D<path>           -> OK / ERR ...
    S1 / S0           -> OK                  (live audio on / off)

Needs bleak (the `device` extra). Everything here takes a connected bleak
client, or anything with the same write_gatt_char / start_notify methods.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from .adpcm import decode_block
from .clock import local_epoch

SERVICE = "7f1e0001-5a1b-4c8e-9d2f-3b6a0c4e8d10"
CTRL = "7f1e0002-5a1b-4c8e-9d2f-3b6a0c4e8d10"
DATA = "7f1e0003-5a1b-4c8e-9d2f-3b6a0c4e8d10"
LIVE = "7f1e0004-5a1b-4c8e-9d2f-3b6a0c4e8d10"
TIMEOUT = 10.0  # seconds without a word from the recorder


class RecorderError(RuntimeError):
    pass


async def find_recorder(timeout: float = 10.0):
    """The first recorder advertising nearby, or None."""
    from bleak import BleakScanner

    return await BleakScanner.find_device_by_filter(
        lambda dev, adv: SERVICE in [u.lower() for u in adv.service_uuids], timeout=timeout
    )


def parse_status(line: str) -> dict:
    """'RECORDER 3 20261020T093000 LISTEN - dropped=0 bat=3.91 pct=65' -> a dict."""
    fields = line.split()
    out = {"clock": fields[2] if len(fields) > 2 else None}
    if len(fields) > 4:
        out["state"], out["file"] = fields[3], fields[4]
    for f in fields[5:]:
        key, _, value = f.partition("=")
        out[key] = value
    return out


class BleRecorder:
    """The protocol over a connected client. Call start() once first."""

    def __init__(self, client, timeout: float = TIMEOUT):
        self.client = client
        self.timeout = timeout
        self._events: asyncio.Queue = asyncio.Queue()  # ("ctrl", line) / ("data", bytes), in order
        self._on_live: Callable[[np.ndarray], None] | None = None
        self.live_lost = 0  # live packets missing (by their counter)
        self._live_seq: int | None = None

    async def start(self) -> None:
        await self.client.start_notify(CTRL, self._ctrl)
        await self.client.start_notify(DATA, self._data)
        await self.client.start_notify(LIVE, self._live)

    def _ctrl(self, _, data: bytearray) -> None:
        self._events.put_nowait(("ctrl", bytes(data).decode(errors="replace").strip()))

    def _data(self, _, data: bytearray) -> None:
        self._events.put_nowait(("data", bytes(data)))

    def _live(self, _, data: bytearray) -> None:
        if self._on_live is None or len(data) < 5:
            return
        seq = data[3]
        if self._live_seq is not None:
            self.live_lost += (seq - self._live_seq - 1) % 256
        self._live_seq = seq
        self._on_live(decode_block(bytes(data)))

    async def _send(self, line: str) -> None:
        await self.client.write_gatt_char(CTRL, line.encode(), response=True)

    async def _next(self) -> tuple[str, object]:
        try:
            return await asyncio.wait_for(self._events.get(), self.timeout)
        except asyncio.TimeoutError:
            raise RecorderError("the recorder stopped answering") from None

    async def _line(self) -> str:
        while True:
            kind, value = await self._next()
            if kind == "ctrl":
                return value

    async def ask(self, line: str) -> str:
        while not self._events.empty():  # leftovers of something abandoned
            self._events.get_nowait()
        await self._send(line)
        return await self._line()

    async def status(self) -> dict:
        reply = await self.ask("?")
        if not reply.startswith("RECORDER"):
            raise RecorderError(f"unexpected answer {reply!r}")
        return parse_status(reply)

    async def set_clock(self, now: datetime | None = None) -> str:
        reply = await self.ask(f"T{local_epoch(now)}")
        if not reply.startswith("OK "):
            raise RecorderError(f"clock not set ({reply!r})")
        return reply[3:]

    async def list_files(self) -> list[tuple[str, int]]:
        """(path on the card, size) of every finished recording."""
        await self.ask("X")  # nothing else running
        await self._send("L")
        files = []
        while True:
            line = await self._line()
            if line == "END":
                return files
            if line.startswith("F "):
                path, _, size = line[2:].rpartition(" ")
                files.append((path, int(size)))
            elif line.startswith("ERR"):
                raise RecorderError(line)

    async def fetch(
        self, path: str, size: int, dest: Path, progress: Callable[[int], None] | None = None
    ) -> None:
        """Download a file to `dest`, resuming a `dest.part` left by an interrupted run."""
        part = dest.with_name(dest.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "ab") as f:
            while True:
                offset = f.tell()
                if await self._fetch_from(path, offset, f, progress):
                    break
        if part.stat().st_size != size:
            raise RecorderError(f"{path}: got {part.stat().st_size} of {size} bytes")
        part.replace(dest)

    async def _fetch_from(self, path, offset, f, progress) -> bool:
        """Receive from `offset` on; False when a packet went missing (ask again from there)."""
        await self._send(f"G{offset} {path}")
        while True:
            kind, value = await self._next()
            if kind == "data":
                at = int.from_bytes(value[:4], "little")
                if at != offset:  # a lost packet: stop, and continue from what we have
                    await self.ask("X")
                    return False
                f.write(value[4:])
                offset += len(value) - 4
                if progress:
                    progress(offset)
            elif value.startswith("DONE"):
                return True
            elif value.startswith("ERR"):
                raise RecorderError(f"{path}: {value}")

    async def delete(self, path: str) -> None:
        reply = await self.ask(f"D{path}")
        if reply != "OK":
            raise RecorderError(f"{path}: {reply}")

    async def live(self, on_audio: Callable[[np.ndarray], None] | None) -> None:
        """Start (with a callback for int16 sample blocks) or stop (None) live audio."""
        self._on_live, self._live_seq = on_audio, None
        reply = await self.ask("S1" if on_audio else "S0")
        if reply != "OK":
            raise RecorderError(f"live audio: {reply}")


async def download_new(
    rec: BleRecorder, inbox: Path, log: Callable[[str], None] = lambda msg: None
) -> list[tuple[str, Path]]:
    """Fetch every finished recording not already in `inbox` (kept as on the card:
    inbox/REC/20261020/093000.WAV); returns (card path, local path) of all the card's files."""
    files = await rec.list_files()
    out = []
    for path, size in files:
        dest = inbox / path
        if not (dest.is_file() and dest.stat().st_size == size):
            log(f"downloading {path} ({size // 1024} KB)")
            await rec.fetch(path, size, dest)
        out.append((path, dest))
    return out
