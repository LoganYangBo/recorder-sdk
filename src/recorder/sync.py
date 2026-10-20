"""Where recordings come from, and when they started.

A source lists audio files with their wall-clock start time. FolderSource is
any folder (phone recordings copied over); DeviceSource is the recorder
itself, mounted over USB as a drive. Device convention:

    <drive>/recorder-device.json     {"id": "..."}  marks the drive as a recorder
    <drive>/REC/20261016T091500.wav  one file per recording, named by its local
                                     start time (from the device's clock)
    <drive>/REC/20261016/091500.wav  the same, for firmware limited to 8.3 names
"""

from __future__ import annotations

import json
import os
import re
import string
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus", ".aac"}
DEVICE_MARKER = "recorder-device.json"

# 20261016T091500, 20261016_091500, 2026-10-16 09.15.00, 2026-10-16T09:15:00+08:00, ...
_STAMP = re.compile(
    r"(?<!\d)(\d{4})-?(\d{2})-?(\d{2})[T_ -]?(\d{2})[:.\-]?(\d{2})[:.\-]?(\d{2})"
    r"(Z|[+-]\d{2}:?\d{2})?(?!\d)"
)


@dataclass
class AudioFile:
    path: Path
    started_at: str | None = None  # local wall-clock time, ISO 8601 (2026-10-16T09:15:00)


class AudioSource(Protocol):
    def list_files(self) -> Iterator[AudioFile]: ...


def _local(dt: datetime) -> str:
    """ISO 8601 local time without offset; aware times are converted to this machine's zone."""
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat()


def parse_started_at(name: str) -> str | None:
    """The start time encoded in a file name, as local ISO 8601, or None."""
    for m in _STAMP.finditer(name):
        y, mo, d, h, mi, s, tz = m.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
        except ValueError:
            continue
        if tz:
            tz = "+00:00" if tz == "Z" else tz if ":" in tz else f"{tz[:3]}:{tz[3:]}"
            dt = datetime.fromisoformat(dt.isoformat() + tz)
        return _local(dt)
    return None


def media_started_at(path: str | Path) -> str | None:
    """The creation time phones write into the file's metadata, or None
    (needs PyAV, installed with the whisper extra)."""
    try:
        import av

        with av.open(str(path)) as c:
            stamp = c.metadata.get("creation_time") or next(
                (s.metadata.get("creation_time") for s in c.streams if s.metadata), None
            )
    except Exception:
        return None
    if not stamp:
        return None
    try:
        return _local(datetime.fromisoformat(stamp.replace("Z", "+00:00")))
    except ValueError:
        return None


def _from_date_folder(path: Path) -> str | None:
    """REC/20261016/091500.wav (or 091500A.wav, a second file in the same second)."""
    if re.fullmatch(r"\d{8}", path.parent.name) and re.fullmatch(r"\d{6}[A-Za-z]?", path.stem):
        return parse_started_at(f"{path.parent.name}T{path.stem[:6]}")
    return None


def started_at_of(path: str | Path) -> str | None:
    """When a recording started: from its name (or date folder), else from its metadata."""
    path = Path(path)
    return parse_started_at(path.name) or _from_date_folder(path) or media_started_at(path)


class FolderSource:
    def __init__(self, root: str | Path, recursive: bool = True):
        self.root = Path(root)
        self.recursive = recursive

    def list_files(self) -> Iterator[AudioFile]:
        pattern = "**/*" if self.recursive else "*"
        for p in sorted(self.root.glob(pattern)):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."):
                yield AudioFile(p, started_at_of(p))


class DeviceSource(FolderSource):
    """The recorder mounted as a drive (see the module docstring)."""

    def __init__(self, root: str | Path):
        root = Path(root)
        info = json.loads((root / DEVICE_MARKER).read_text() or "{}")
        self.id = str(info.get("id") or root.name)
        self.drive = root
        super().__init__(root / "REC" if (root / "REC").is_dir() else root)


def _mount_points() -> list[Path]:
    if sys.platform == "win32":
        return [Path(f"{c}:\\") for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    roots = [Path("/Volumes")] if sys.platform == "darwin" else []
    user = os.environ.get("USER", "")
    roots += [Path("/media") / user, Path("/run/media") / user, Path("/media"), Path("/mnt")]
    found = []
    for r in roots:
        try:
            found += [p for p in r.iterdir() if p.is_dir()]
        except OSError:
            pass
    return found


def find_devices(mounts: list[Path] | None = None) -> list[DeviceSource]:
    """Every mounted drive that carries the recorder's marker file."""
    out = []
    for m in _mount_points() if mounts is None else mounts:
        try:
            if (m / DEVICE_MARKER).is_file():
                out.append(DeviceSource(m))
        except (OSError, ValueError):
            pass
    return out
