from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import Segment, Transcript, Word

if TYPE_CHECKING:
    import numpy as np


def _add_nvidia_dll_dirs() -> None:
    """On Windows, make DLLs from the nvidia-cublas-cu12 / nvidia-cudnn-cu12
    pip packages loadable. CTranslate2 loads cuBLAS/cuDNN lazily by name, so
    their bin folders must be on PATH (Linux wheels set rpath; Windows doesn't).
    """
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class Transcriber(Protocol):
    def transcribe(self, path: str | Path) -> Transcript: ...


class WhisperTranscriber:
    """faster-whisper wrapper. The model is loaded lazily on first use.

    device="auto" picks CUDA when available. compute_type="default" lets
    CTranslate2 choose (float16 on GPU, int8 on CPU).

    initial_prompt / hotwords help with names and jargon Whisper would
    otherwise mishear, e.g. hotwords="PyTorch Kubernetes".
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = True,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self._model = None

    @property
    def params(self) -> dict:
        p = {
            "engine": "faster-whisper",
            "model": self.model_name,
            "language": self.language,
            "vad_filter": self.vad_filter,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "initial_prompt": self.initial_prompt,
            "hotwords": self.hotwords,
        }
        return {k: v for k, v in p.items() if v is not None}

    def _load(self):
        if self._model is None:
            _add_nvidia_dll_dirs()
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise ImportError(
                    "faster-whisper is not installed: pip install 'recorder[whisper]'"
                ) from e
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, path: str | Path | np.ndarray) -> Transcript:
        """Transcribe an audio file, or 16 kHz mono float32 samples."""
        model = self._load()
        audio = path if hasattr(path, "dtype") else str(path)
        segments, info = model.transcribe(
            audio,
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
        )
        # segments is a generator; transcription actually runs here
        segs = [
            Segment(
                s.start,
                s.end,
                s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=[Word(w.start, w.end, w.word.strip(), w.probability) for w in s.words or []],
            )
            for s in segments
        ]
        return Transcript(
            language=info.language, duration=info.duration, segments=segs, params=self.params
        )

    # ---- for correcting single words ----

    def alternatives(
        self,
        audio: np.ndarray,
        spans: list[tuple[float, float]],
        language: str | None = None,
        temperatures: tuple[float, ...] = (0.4, 0.7, 1.0, 1.0, 1.2),
    ) -> list[list[str]]:
        """Other words Whisper might hear in each (start, end) span (seconds)
        of `audio` (16 kHz samples, typically one segment): re-decodes it with
        sampling at each temperature and collects, per span, the words that
        overlap it. Words may repeat; the more often, the likelier. One pass
        over the audio serves every span in it."""
        model = self._load()
        out: list[list[str]] = [[] for _ in spans]
        for t in temperatures:
            segments, _ = model.transcribe(
                audio,
                language=language or self.language,
                temperature=t,
                beam_size=1,
                best_of=1,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
                hotwords=self.hotwords,
            )
            for seg in segments:
                for w in seg.words or []:
                    for found, (start, end) in zip(out, spans):
                        if min(w.end, end) - max(w.start, start) > 0.3 * (end - start):
                            found.append(w.word.strip())
        return out

    def word_rank(self, word: str) -> int | None:
        """How common an English word is, from Whisper's tokenizer: words that
        are a single token rank by token id (lower = more common); others None."""
        tok = getattr(self._load(), "hf_tokenizer", None)
        if tok is None:
            return 0
        ids = tok.encode(" " + word, add_special_tokens=False).ids
        return ids[0] if len(ids) == 1 else None
