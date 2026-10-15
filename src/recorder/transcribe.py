from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Segment, Transcript


class Transcriber(Protocol):
    def transcribe(self, path: str | Path) -> Transcript: ...


class WhisperTranscriber:
    """faster-whisper wrapper. The model is loaded lazily on first use.

    device="auto" picks CUDA when available. compute_type="default" lets
    CTranslate2 choose (float16 on GPU, int8 on CPU).
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self._model = None

    def _load(self):
        if self._model is None:
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

    def transcribe(self, path: str | Path) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(
            str(path),
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
        )
        # segments is a generator; transcription actually runs here
        segs = [Segment(s.start, s.end, s.text.strip()) for s in segments]
        return Transcript(language=info.language, duration=info.duration, segments=segs)
