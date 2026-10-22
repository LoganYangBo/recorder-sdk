"""Set the recorder's clock over USB serial.

The recorder (firmware/recorder) has no battery-backed clock, so it learns the time from the
computer whenever it is plugged in, and keeps it while it stays switched on. One line each
way, 115200 baud:

    ?            -> RECORDER 3 <20261020T093000|unset> <LISTEN|REC|PAUSED|FAILED> <file|->
                    dropped=<n> level=<sound level now> floor=<noise floor> bat=<V> pct=<%>
    T<seconds>   -> OK 20261020T093000

<seconds> is the local wall-clock time counted as if it were UTC, so the device names files in
local time without knowing the time zone. Needs pyserial (the `device` extra). Over
Bluetooth the same commands work, and more: see ble.py.
"""

from __future__ import annotations

import calendar
import time
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
    settle: float = 0.5,
    log: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Send the time to every plugged-in recorder; returns (port, the time it now reports).

    Ports that don't answer like a recorder are left alone. `log` hears what each port said."""
    if open_port is None:
        import serial

        def open_port(name):
            return serial.Serial(name, BAUD, timeout=1)

    log = log or (lambda msg: None)
    done = []
    for name in recorder_ports() if ports is None else ports:
        try:
            with open_port(name) as port:
                time.sleep(settle)  # the board may miss what arrives right as the port opens
                for _ in range(3):
                    hello = _ask(port, "?")
                    if hello.startswith("RECORDER"):
                        break
                else:
                    log(f"{name}: not a recorder (answered {hello!r})")
                    continue
                reply = _ask(port, f"T{local_epoch(now)}")
                if not reply.startswith("OK "):  # busy switching files: did it take?
                    fields = _ask(port, "?").split()
                    if len(fields) > 2 and fields[0] == "RECORDER" and fields[2] != "unset":
                        reply = f"OK {fields[2]}"
        except OSError as e:  # busy (the Arduino serial monitor has it), or unplugged meanwhile
            log(f"{name}: can't open ({e})")
            continue
        if reply.startswith("OK "):
            done.append((name, reply[3:]))
        else:
            log(f"{name}: didn't take the time (answered {reply!r})")
    return done
