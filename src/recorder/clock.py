"""Set the recorder's clock over USB serial.

The recorder (firmware/recorder) has no battery-backed clock, so it learns the time from the
computer whenever it is plugged in, and keeps it while it stays switched on. One line each
way, 115200 baud:

    ?            -> RECORDER 1 <20261020T093000|unset> <REC|PAUSED|FAILED> <file> dropped=<n>
    T<seconds>   -> OK 20261020T093000

<seconds> is the local wall-clock time counted as if it were UTC, so the device names files in
local time without knowing the time zone. Needs pyserial (the `device` extra).
"""

from __future__ import annotations

import calendar
from datetime import datetime
from typing import Callable

SEEED_VID = 0x2886  # USB vendor id of Seeed boards, the XIAO among them
BAUD = 115200


def local_epoch(now: datetime | None = None) -> int:
    """Local wall-clock time as seconds since 1970, as if it were UTC."""
    return calendar.timegm((now or datetime.now()).timetuple())


def recorder_ports() -> list[str]:
    """Serial ports of plugged-in Seeed boards."""
    from serial.tools import list_ports

    return [p.device for p in list_ports.comports() if p.vid == SEEED_VID]


def _ask(port, line: str) -> str:
    port.reset_input_buffer()
    port.write(f"{line}\n".encode())
    return port.readline().decode(errors="replace").strip()


def set_clock(
    ports: list[str] | None = None,
    now: datetime | None = None,
    open_port: Callable | None = None,
) -> list[tuple[str, str]]:
    """Send the time to every plugged-in recorder; returns (port, the time it now reports).

    Ports that don't answer like a recorder are left alone."""
    if open_port is None:
        import serial

        def open_port(name):
            return serial.Serial(name, BAUD, timeout=2)

    done = []
    for name in recorder_ports() if ports is None else ports:
        try:
            with open_port(name) as port:
                if not _ask(port, "?").startswith("RECORDER"):
                    continue
                reply = _ask(port, f"T{local_epoch(now)}")
        except OSError:  # busy (the Arduino serial monitor has it), or unplugged meanwhile
            continue
        if reply.startswith("OK "):
            done.append((name, reply[3:]))
    return done
