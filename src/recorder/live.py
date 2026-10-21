"""Live transcription of a growing audio stream (the browser microphone, the recorder).

Audio arrives in small chunks and is cut at pauses in speech: once someone stops
for `pause` seconds, everything since the last cut is transcribed as one piece
and committed (and speaker-labelled). Cutting in silence keeps words whole, and
each piece is a phrase or a few sentences that Whisper sees in one go, with the
text before it as context. Until the pause comes, the piece in progress is
re-transcribed every `min_new` seconds and shown as partial text. finish()
commits whatever is left; save() writes the audio to a WAV file and stores it
as a normal recording.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .ingest import has_speech
from .models import Segment, Transcript
from .speaker import SAMPLE_RATE
from .store import Store

FRAME = SAMPLE_RATE // 50  # 20 ms frames for finding pauses


class LiveSession:
    def __init__(
        self,
        transcriber,
        labeler=None,
        pause: float = 0.6,
        max_piece: float = 20.0,
        min_new: float = 1.0,
    ):
        """pause: this much quiet ends a piece. max_piece: a piece with no pause is cut
        (at its quietest moment) after this many seconds. min_new: refresh the partial
        text after this much new audio.
        """
        self.transcriber = transcriber
        self.labeler = labeler
        self.pause = pause
        self.max_piece = max_piece
        self.min_new = min_new
        self.started_at = datetime.now().astimezone()
        self._chunks: list[np.ndarray] = []
        self._n = 0  # samples received
        self._audio = np.zeros(0, np.float32)
        self._rms = np.zeros(0, np.float32)  # loudness of each whole frame so far
        self.committed: list[Segment] = []
        self.committed_until = 0.0  # seconds
        self.partial = ""
        self.language = None
        self._last_pass = 0  # samples at the last partial transcription
        params = inspect.signature(transcriber.transcribe).parameters
        self._context_ok = "language" in params and "prompt" in params
        self.lock = threading.Lock()

    @property
    def duration(self) -> float:
        return self._n / SAMPLE_RATE

    def _all_audio(self) -> np.ndarray:
        if self._chunks:
            self._audio = np.concatenate([self._audio, *self._chunks])
            self._chunks = []
            whole = len(self._audio) // FRAME
            if whole > len(self._rms):
                x = self._audio[len(self._rms) * FRAME : whole * FRAME].reshape(-1, FRAME)
                self._rms = np.concatenate([self._rms, np.sqrt(np.mean(x * x, axis=1))])
        return self._audio

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Add 16 kHz mono float32 samples; returns segments newly committed."""
        with self.lock:
            self._chunks.append(np.asarray(samples, np.float32))
            self._n += len(samples)
            self._all_audio()
            done = []
            while (cut := self._find_cut()) is not None:
                done += self._commit(cut)
            if self._n - self._last_pass >= self.min_new * SAMPLE_RATE:
                self._refresh_partial()
            return done

    def finish(self) -> list[Segment]:
        """Commit whatever is left; returns the segments committed now."""
        with self.lock:
            self._all_audio()
            done = self._commit(len(self._audio))
            self.partial = ""
            return done

    def _find_cut(self) -> int | None:
        """Where (a sample) the piece in progress ends, if it has ended."""
        first = int(self.committed_until * SAMPLE_RATE) // FRAME
        rms = self._rms[first:]
        if not len(rms):
            return None
        # background noise: the quietest half second of the last minute
        recent = self._rms[-3000:]
        blocks = recent[: len(recent) // 25 * 25].reshape(-1, 25)
        floor = float(np.median(blocks, axis=1).min()) if len(blocks) else float(recent.min())
        quiet = rms < max(floor * 2.5, 1e-4)
        need = int(self.pause * SAMPLE_RATE / FRAME)
        longest = int(self.max_piece * SAMPLE_RATE / FRAME)
        talk = np.flatnonzero(~quiet)
        if not len(talk):  # quiet so far: hand it over now and then (Whisper has the last word)
            return (first + len(rms)) * FRAME if len(rms) >= longest else None
        run = 0
        for i in range(talk[0], len(rms)):
            run = run + 1 if quiet[i] else 0
            if run == need:  # a pause after speech: cut in the middle of it
                return (first + i - need // 2) * FRAME
        if len(rms) >= longest:  # no pause: cut at the quietest moment of the second half
            half = longest // 2
            smooth = np.convolve(rms[half:longest], np.ones(10) / 10, mode="same")
            return (first + half + int(np.argmin(smooth))) * FRAME
        return None

    def _transcribe(self, audio: np.ndarray) -> Transcript:
        if not self._context_ok:
            return self.transcriber.transcribe(audio)
        before = " ".join(s.text for s in self.committed[-3:])[-200:] or None
        return self.transcriber.transcribe(audio, language=self.language, prompt=before)

    def _commit(self, cut: int) -> list[Segment]:
        offset = self.committed_until
        piece = self._audio[int(offset * SAMPLE_RATE) : cut]
        self.committed_until = cut / SAMPLE_RATE
        self._last_pass = 0  # the partial text is out of date
        self.partial = ""
        if len(piece) < 0.3 * SAMPLE_RATE:
            return []
        t = self._transcribe(piece)
        self.language = self.language or t.language
        done = [s for s in t.segments if has_speech([s])]  # not what Whisper makes up in quiet
        if done and self.labeler is not None:
            self.labeler.label_audio(piece, Transcript(t.language, len(piece) / SAMPLE_RATE, done))
        for s in done:
            s.start += offset
            s.end += offset
            for w in s.words:
                w.start += offset
                w.end += offset
        self.committed.extend(done)
        return done

    def _refresh_partial(self) -> None:
        self._last_pass = self._n
        piece = self._audio[int(self.committed_until * SAMPLE_RATE) :]
        if len(piece) < 0.3 * SAMPLE_RATE:
            self.partial = ""
            return
        t = self._transcribe(piece)
        self.language = self.language or t.language
        self.partial = " ".join(s.text for s in t.segments)

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
        return store.save(
            path,
            digest,
            t,
            started_at=self.started_at.replace(tzinfo=None).isoformat(timespec="seconds"),
        )
