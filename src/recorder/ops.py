"""Speaker and unsure-word operations shared by the CLI and the web UI."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
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


def suggest_words(
    store: Store,
    transcriber,
    recording_id: int,
    lock=None,
    progress: Callable[[int, int], None] | None = None,
    aligner=None,
) -> int:
    """Work out correction candidates for every unsure word of a recording
    not analysed yet, and store them; returns how many words were done.
    With an `aligner` (align.WordAligner), the segments holding those words
    are first force-aligned for precise word times. `lock` (if given) is
    held around each model call; `progress(done, total)` is called after
    each segment."""
    from . import learn
    from .align import refine
    from .speaker import SAMPLE_RATE, load_audio

    rec = store.recording(recording_id)
    todo = store.unsure_words(recording_id, align=aligner is not None)
    if rec is None or not todo:
        return 0
    by_segment: dict[tuple[int, float, float], list] = {}
    for seg_id, s0, s1, w in todo:
        by_segment.setdefault((seg_id, s0, s1), []).append(w)
    audio = load_audio(rec.path)
    done = 0
    for (seg_id, s0, s1), words in by_segment.items():
        if aligner is not None and not store.segment_aligned(seg_id):
            # a little extra audio either side, in case the segment cuts a word
            a0, a1 = max(0.0, s0 - 0.3), min(len(audio) / SAMPLE_RATE, s1 + 0.3)
            seg_words = store.segment_words(seg_id)
            try:
                with lock or contextlib.nullcontext():
                    found = aligner.align(
                        audio[int(a0 * SAMPLE_RATE) : int(a1 * SAMPLE_RATE)],
                        [w.word for w in seg_words],
                    )
                found = [(a0 + t[0], a0 + t[1]) if t else None for t in found]
                times = refine([(w.start, w.end) for w in seg_words], found)
            except Exception:  # e.g. more letters than audio frames: keep Whisper's times
                times = [None] * len(seg_words)
            store.set_word_times({w.id: t for w, t in zip(seg_words, times)})
            new = {w.id: t for w, t in zip(seg_words, times) if t}
            for w in words:
                if w.id in new:
                    w.start, w.end = new[w.id]
        words = [w for w in words if store.candidates(w.id) is None]
        if words:
            clip = audio[int(s0 * SAMPLE_RATE) : int(s1 * SAMPLE_RATE)]
            spans = [(w.start - s0, w.end - s0) for w in words]
            with lock or contextlib.nullcontext():
                alts = transcriber.alternatives(clip, spans, language=rec.language)
                found = [
                    learn.candidates(w.original or w.word, a, transcriber.word_rank)
                    for w, a in zip(words, alts)
                ]
            for w, cands in zip(words, found):
                store.set_candidates(w.id, cands)
        done += len(by_segment[(seg_id, s0, s1)])
        if progress:
            progress(done, len(todo))
    return done
