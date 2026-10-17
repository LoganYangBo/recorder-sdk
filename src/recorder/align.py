"""Word-level forced alignment, for precise word start/end times.

Whisper's word timestamps come from its attention and are often 0.1-0.3 s
off, so a clip cut from them can cut into the word or include its
neighbours. Forced alignment matches the known text against a wav2vec2 CTC
model's per-frame (20 ms) letter probabilities instead.

Uses torchaudio's MMS_FA bundle (installed with the `speaker` extra); the
model weights (~1.2 GB) are downloaded on first use.
"""

from __future__ import annotations

import re

import numpy as np

SAMPLE_RATE = 16000
MAX_SHIFT = 1.0  # seconds; an aligned word further than this from Whisper's is not trusted


def letters(word: str) -> str:
    """What the aligner can match: lowercase a-z and apostrophes ('Friday.' -> 'friday').
    Empty for words like '20' or '%', which are aligned as a wildcard."""
    return re.sub(r"[^a-z']", "", word.lower()).strip("'")


def word_times(
    emission, n_samples: int, words: list[str], tokenizer, aligner
) -> list[tuple[float, float] | None]:
    """(start, end) in seconds for each word, from a CTC `emission` (frames x
    tokens log-probabilities) of `n_samples` of audio; None for words that
    could only be aligned as a wildcard."""
    tokens = [letters(w) or "*" for w in words]
    spans = aligner(emission, tokenizer(tokens))
    sec_per_frame = n_samples / emission.shape[0] / SAMPLE_RATE
    return [
        None if tok == "*" else (s[0].start * sec_per_frame, s[-1].end * sec_per_frame)
        for tok, s in zip(tokens, spans)
    ]


def refine(
    whisper: list[tuple[float, float]], aligned: list[tuple[float, float] | None]
) -> list[tuple[float, float] | None]:
    """Aligned times where they are plausible (near Whisper's), else None."""
    out = []
    for (ws, we), a in zip(whisper, aligned):
        ok = a is not None and abs((a[0] + a[1]) / 2 - (ws + we) / 2) <= MAX_SHIFT
        out.append(a if ok else None)
    return out


class WordAligner:
    """Forced aligner for English text (torchaudio MMS_FA)."""

    def __init__(self, device: str = "auto"):
        import torch
        from torchaudio.pipelines import MMS_FA

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = MMS_FA.get_model(with_star=True).to(device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()

    def align(self, audio: np.ndarray, words: list[str]) -> list[tuple[float, float] | None]:
        """(start, end) of each of `words` within `audio` (16 kHz), or None."""
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            emission, _ = self.model(x.unsqueeze(0).to(self.device))
        return word_times(emission[0].cpu(), len(audio), words, self.tokenizer, self.aligner)
