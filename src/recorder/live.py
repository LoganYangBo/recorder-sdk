"""Live transcription of a growing audio stream (e.g. the browser microphone).

Audio arrives in small chunks. Each feed() re-transcribes only the part not
yet committed (the "tail"); segments that end well before the end of the tail
are final and get committed (and speaker-labelled), the rest is returned as
partial text that may still change. finish() commits everything, writes the
audio to a WAV file and stores it as a normal recording.
"""

from __future__ import annotations

import hashlib
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import Segment, Transcript
from .speaker import SAMPLE_RATE
from .store import Store


class LiveSession:
    def __init__(
        self,
        transcriber,
        labeler=None,
        settle: float = 1.5,
        max_tail: float = 25.0,
        min_new: float = 1.0,
    ):
        """settle: a segment counts as final once the audio runs this many
        seconds past its end. max_tail: never keep more than this many
        seconds uncommitted. min_new: skip transcribing until this much new
        audio has arrived since the last pass.
        """
        self.transcriber = transcriber
        self.labeler = labeler
        self.settle = settle
        self.max_tail = max_tail
        self.min_new = min_new
        self.started_at = datetime.now().astimezone()
        self._chunks: list[np.ndarray] = []
        self._n = 0  # samples received
        self._audio = np.zeros(0, np.float32)
        self.committed: list[Segment] = []
        self.committed_until = 0.0  # seconds
        self.partial = ""
        self.language = None
        self._last_pass = 0  # samples at the last transcription
        self.lock = threading.Lock()

    @property
    def duration(self) -> float:
        return self._n / SAMPLE_RATE

    def _all_audio(self) -> np.ndarray:
        if self._chunks:
            self._audio = np.concatenate([self._audio, *self._chunks])
            self._chunks = []
        return self._audio

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Add 16 kHz mono float32 samples; returns segments newly committed."""
        with self.lock:
            self._chunks.append(np.asarray(samples, np.float32))
            self._n += len(samples)
            if self._n - self._last_pass < self.min_new * SAMPLE_RATE:
                return []
            return self._pass(final=False)

    def finish(self) -> list[Segment]:
        """Commit whatever is left; returns the segments committed now."""
        with self.lock:
            return self._pass(final=True)

    def _pass(self, final: bool) -> list[Segment]:
        audio = self._all_audio()
        self._last_pass = self._n
        offset = self.committed_until
        tail = audio[int(offset * SAMPLE_RATE) :]
        tail_dur = len(tail) / SAMPLE_RATE
        if tail_dur < 0.3:
            self.partial = ""
            return []
        t = self.transcriber.transcribe(tail)
        self.language = self.language or t.language
        segs = t.segments
        if final:
            done, pending = segs, []
        else:
            done = [s for s in segs if s.end <= tail_dur - self.settle]
            pending = segs[len(done) :]
            if tail_dur > self.max_tail and pending:  # don't let the tail grow forever
                keep = 1 if len(segs) > 1 else 0
                done, pending = segs[: len(segs) - keep], segs[len(segs) - keep :]
        if done and self.labeler is not None:
            self.labeler.label_audio(tail, Transcript(t.language, tail_dur, done))
        for s in done:
            s.start += offset
            s.end += offset
            for w in s.words:
                w.start += offset
                w.end += offset
        self.committed.extend(done)
        if final:
            self.committed_until = self.duration
        elif done:
            self.committed_until = done[-1].end
        elif not segs and tail_dur > self.max_tail:
            self.committed_until = self.duration - self.settle  # long silence: drop it
        self.partial = " ".join(s.text for s in pending)
        return done

    def save(self, store: Store, library: Path, params: dict | None = None) -> int:
        """Write the audio to `library` as WAV and store the transcript; returns the id."""
        library.mkdir(parents=True, exist_ok=True)
        path = library / f"live-{self.started_at:%Y%m%d-%H%M%S}.wav"
        pcm = (np.clip(self._all_audio(), -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        t = Transcript(
            self.language or "unknown",
            self.duration,
            self.committed,
            params={**(params or {}), "live": True},
        )
        return store.save(path, digest, t, started_at=self.started_at.isoformat(timespec="seconds"))
