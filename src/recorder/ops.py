"""Speaker operations shared by the CLI and the web UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Transcript
from .pipeline import Pipeline
from .store import Store

if TYPE_CHECKING:
    import numpy as np

    from .speaker import Embedder, VoiceprintLabeler

RESERVED_NAMES = {"", "other"}


def voiceprint_labeler(
    store: Store, embedder: Embedder, threshold: float
) -> VoiceprintLabeler | None:
    """Labeler over every enrolled voice, or None if nobody is enrolled."""
    import numpy as np

    from .speaker import VoiceprintLabeler

    vps = store.voiceprints()
    if not vps:
        return None
    stale = [name for name, (model, _) in vps.items() if model != embedder.name]
    if stale:
        raise ValueError(f"voiceprints {stale} were made with another model; enroll them again")
    voices = {name: np.frombuffer(blob, dtype=np.float32) for name, (_, blob) in vps.items()}
    return VoiceprintLabeler(embedder, voices, threshold)


def segment_audio(store: Store, recording_id: int, indices: list[int]) -> list[np.ndarray]:
    """Audio of a recording's segments, numbered from 1 in `segments()` order."""
    from .speaker import SAMPLE_RATE, load_audio

    rec = store.recording(recording_id)
    if rec is None:
        raise ValueError(f"no recording {recording_id}")
    segments = store.segments(recording_id)
    if not indices or not all(1 <= i <= len(segments) for i in indices):
        raise ValueError(f"segment numbers must be 1..{len(segments)}")
    audio = load_audio(rec.path)
    return [
        audio[int(segments[i - 1].start * SAMPLE_RATE) : int(segments[i - 1].end * SAMPLE_RATE)]
        for i in indices
    ]


def enroll_voice(store: Store, embedder: Embedder, name: str, audios: list[np.ndarray]) -> None:
    """Save (or replace) the voiceprint for `name`."""
    from .speaker import enroll

    name = name.strip()
    if name in RESERVED_NAMES:
        raise ValueError(f"'{name}' can't be used as a name")
    vec = enroll(embedder, audios)
    store.save_voiceprint(name, embedder.name, vec.tobytes())


def relabel(
    store: Store, labeler: VoiceprintLabeler, recording_ids: list[int], rescore: bool
) -> None:
    """Re-label recordings. rescore=False only re-applies the threshold to the
    stored best matches (instant); rescore=True re-embeds the audio.
    """
    pipe = Pipeline(None, store, labeler)
    for rid in recording_ids:
        if rescore:
            pipe.relabel(rid)
        else:
            segs = store.segments(rid)
            labeler.relabel(Transcript("", 0.0, segs))
            store.set_speakers(rid, segs)
