"""Synthesised "standard" pronunciations in an American female or male
voice, as an alternative to dictionary recordings (whose speaker varies by
word). Uses Kokoro-82M (Apache 2.0) through kokoro-onnx: `pip install -e
'.[tts]'`; the model files (~340 MB) are downloaded on first use.
"""

from __future__ import annotations

import io
import urllib.request
import wave
from pathlib import Path

import numpy as np

VOICES = {"female": "af_heart", "male": "am_michael"}  # Kokoro, American English
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def trim(samples: np.ndarray, rate: int, threshold: float = 0.02, pad: float = 0.05) -> np.ndarray:
    """Cut leading and trailing silence, keeping `pad` seconds."""
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    loud = np.flatnonzero(np.abs(samples) > threshold * max(peak, 1e-6))
    if not len(loud):
        return samples
    p = int(pad * rate)
    return samples[max(0, loud[0] - p) : loud[-1] + 1 + p]


def download(folder: Path) -> list[Path]:
    """The Kokoro model files in `folder`, downloaded if missing."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in MODEL_FILES:
        path = folder / name
        if not path.is_file():
            print(f"downloading {name} for the female / male voices...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL + name, tmp)
            tmp.replace(path)
        paths.append(path)
    return paths


class KokoroVoices:
    """Say single words in an American English female or male voice."""

    def __init__(self, folder: Path):
        from kokoro_onnx import Kokoro  # import first: no download without the package

        model, voices = download(folder)
        self.kokoro = Kokoro(str(model), str(voices))

    def say(self, word: str, voice: str) -> tuple[np.ndarray, int]:
        samples, rate = self.kokoro.create(word, voice=VOICES[voice], speed=0.9, lang="en-us")
        return np.asarray(samples, np.float32), rate
