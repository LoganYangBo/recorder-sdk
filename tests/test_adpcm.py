import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from recorder.adpcm import decode_block, samples_per_block

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware" / "recorder"


def ima_wav(blocks: list[bytes], block_bytes: int = 256) -> bytes:
    """A WAV file as the firmware writes it (format 0x11, 60-byte header)."""
    spb = samples_per_block(block_bytes)
    data = b"".join(blocks)
    fmt = struct.pack(
        "<HHIIHHHH", 0x11, 1, 16000, 16000 * block_bytes // spb, block_bytes, 4, 2, spb
    )
    return (
        b"RIFF" + struct.pack("<I", 52 + len(data)) + b"WAVEfmt " + struct.pack("<I", 20) + fmt
        + b"fact" + struct.pack("<II", 4, spb * len(blocks))
        + b"data" + struct.pack("<I", len(data)) + data
    )  # fmt: skip


def test_samples_per_block():
    assert samples_per_block(256) == 505
    assert samples_per_block(240) == 473


def test_decodes_like_ffmpeg(tmp_path):
    av = pytest.importorskip("av")
    rng = np.random.default_rng(1)
    blocks = []
    for _ in range(6):  # any bytes are a valid block; keep the index in range
        b = bytearray(rng.integers(0, 256, 256, dtype=np.uint8).tobytes())
        b[2], b[3] = rng.integers(0, 89), 0
        blocks.append(bytes(b))
    path = tmp_path / "x.wav"
    path.write_bytes(ima_wav(blocks))
    with av.open(str(path)) as c:
        theirs = np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)])
    ours = np.concatenate([decode_block(b) for b in blocks])
    assert theirs.dtype == np.int16
    np.testing.assert_array_equal(ours, theirs[: len(ours)])


@pytest.mark.skipif(not (shutil.which("cc") or shutil.which("gcc")), reason="needs a C compiler")
def test_firmware_encoder_round_trip(tmp_path):
    """Compile the firmware's encoder and check our decoder gets the sound back."""
    src = tmp_path / "enc.cpp"
    src.write_text(  # files, not stdin/stdout: those are text mode on Windows
        '#include "adpcm.h"\n#include <stdio.h>\n'
        "int main(int argc, char** argv) { int16_t x[505]; uint8_t out[256]; Adpcm s;\n"
        '  FILE* in = fopen(argv[1], "rb"); FILE* o = fopen(argv[2], "wb");\n'
        "  while (fread(x, 2, 505, in) == 505) { adpcmBlock(s, x, out, 256, 0);"
        " fwrite(out, 1, 256, o); }\n  fclose(o); }\n"
    )
    exe = tmp_path / "enc"
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        pytest.skip("needs a C++ compiler")
    subprocess.run([cc, "-I", str(FIRMWARE), "-o", str(exe), str(src)], check=True)
    t = np.arange(505 * 40) / 16000
    x = (8000 * np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 3 * t)).astype("<i2")
    (tmp_path / "x.raw").write_bytes(x.tobytes())
    subprocess.run([str(exe), str(tmp_path / "x.raw"), str(tmp_path / "x.adpcm")], check=True)
    enc = (tmp_path / "x.adpcm").read_bytes()
    assert len(enc) == 256 * 40
    y = np.concatenate([decode_block(enc[i : i + 256]) for i in range(0, len(enc), 256)])
    noise = np.sqrt(np.mean((y.astype(float) - x) ** 2))
    assert noise < 0.03 * np.sqrt(np.mean(x.astype(float) ** 2))  # SNR above 30 dB
