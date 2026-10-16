"""Speaker labelling: who said each segment — "me" (the wearer), a person
you have enrolled by name, or "other".

Enroll yourself once from a recording of only your own voice
(`recorder enroll`), and other people from segments of existing recordings
(`recorder enroll --name Alice --recording 3 --segments 4,7,12`). Each
voiceprint is an averaged speaker embedding stored in the database. Each
transcribed segment is embedded and compared with every voiceprint by cosine
similarity; the best match and score are kept on the segment so the threshold
can be tuned later with `recorder relabel` without re-embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import Transcript

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16000


class SpeakerLabeler(Protocol):
    def label(self, path: str | Path, transcript: Transcript) -> Transcript: ...


class NoopLabeler:
    """Leaves every segment's speaker as None."""

    def label(self, path: str | Path, transcript: Transcript) -> Transcript:
        return transcript


class Embedder(Protocol):
    name: str  # stored with the voiceprint; embeddings of different models don't compare

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 samples -> 1-D speaker embedding."""
        ...


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN trained on VoxCeleb (192-dim). The weights
    (~80 MB) download from Hugging Face on first use; no token needed.
    """

    name = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str = "auto", savedir: str | Path | None = None):
        self.device = device
        self.savedir = Path(savedir) if savedir else Path.home() / ".cache" / "recorder" / "ecapa"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
            except ImportError as e:
                raise ImportError(
                    "speaker labelling needs speechbrain: pip install 'recorder[speaker]'"
                ) from e
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = EncoderClassifier.from_hparams(
                source=self.name,
                savedir=str(self.savedir),
                run_opts={"device": device},
                # copy rather than symlink from the HF cache: symlinks need admin on Windows
                local_strategy=LocalStrategy.COPY,
            )
        return self._model

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        model = self._load()
        with torch.no_grad():
            emb = model.encode_batch(torch.from_numpy(audio).unsqueeze(0))
        return emb.squeeze().cpu().numpy()


def load_audio(path: str | Path) -> np.ndarray:
    from faster_whisper import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _normalize(v: np.ndarray) -> np.ndarray:
    import numpy as np

    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def enroll(embedder: Embedder, audios: list[np.ndarray], window: float = 3.0) -> np.ndarray:
    """Average embedding of recordings of one person speaking, cut into
    `window`-second chunks (shorter clips of at least 1 s are used whole).
    Near-silent chunks are skipped. Returns a unit vector.
    """
    import numpy as np

    n = int(window * SAMPLE_RATE)
    chunks = []
    for a in audios:
        if len(a) < n:
            if len(a) >= SAMPLE_RATE:
                chunks.append(a)
        else:
            chunks.extend(a[i : i + n] for i in range(0, len(a) - n + 1, n))
    if not chunks:
        raise ValueError("need at least 1 s of audio to enroll")
    rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
    voiced = [c for c, r in zip(chunks, rms) if r >= 0.25 * np.median(rms)]
    return _normalize(np.mean([_normalize(embedder.embed(c)) for c in voiced], axis=0))


class VoiceprintLabeler:
    """Labels each segment with the enrolled person whose voiceprint it is
    most similar to (cosine), or "other" when even the best match is below
    `threshold`. The wearer is enrolled as "me". The best match and its score
    are kept on the segment (speaker_match / speaker_score), so the threshold
    can be changed later without re-embedding.

    Segments shorter than `min_duration` are widened around their centre,
    since embeddings of very short clips are unreliable.
    """

    def __init__(
        self,
        embedder: Embedder,
        voiceprints: dict[str, np.ndarray],
        threshold: float = 0.3,
        min_duration: float = 1.5,
    ):
        import numpy as np

        if not voiceprints:
            raise ValueError("no voiceprints enrolled")
        self.embedder = embedder
        self.names = list(voiceprints)
        self.matrix = np.stack([_normalize(v) for v in voiceprints.values()])
        self.threshold = threshold
        self.min_duration = min_duration

    def _clip(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        pad = max(0.0, self.min_duration - (end - start)) / 2
        a = max(0, int((start - pad) * SAMPLE_RATE))
        b = min(len(audio), int((end + pad) * SAMPLE_RATE))
        return audio[a:b]

    def label(self, path: str | Path, transcript: Transcript) -> Transcript:
        audio = load_audio(path)
        for s in transcript.segments:
            clip = self._clip(audio, s.start, s.end)
            if len(clip) < SAMPLE_RATE // 4:  # under 0.25 s: nothing to go on
                s.speaker = s.speaker_match = s.speaker_score = None
                continue
            scores = self.matrix @ _normalize(self.embedder.embed(clip))
            best = int(scores.argmax())
            s.speaker_match, s.speaker_score = self.names[best], float(scores[best])
        return self.relabel(transcript)

    def relabel(self, transcript: Transcript) -> Transcript:
        """Apply the threshold to the best matches already on the segments."""
        for s in transcript.segments:
            if s.speaker_score is not None:
                ok = s.speaker_score >= self.threshold
                s.speaker = s.speaker_match if ok else "other"
        return transcript
