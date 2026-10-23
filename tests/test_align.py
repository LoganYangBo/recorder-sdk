import numpy as np
import pytest

from recorder import Segment, Store, Transcript, Word, ops
from recorder import speaker as speaker_mod
from recorder.align import letters, refine, word_times


def test_letters():
    assert letters("Friday.") == "friday" and letters("don't") == "don't" and letters("20%") == ""


def test_word_times_from_emission():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from torchaudio.pipelines import MMS_FA

    d = MMS_FA.get_dict()
    # 1 s of audio = 50 frames of 20 ms: silence, "hi" at frames 10-19, silence, "yo" at 30-39
    path = ["-"] * 10 + ["h"] * 5 + ["i"] * 5 + ["-"] * 10 + ["y"] * 5 + ["o"] * 5 + ["-"] * 10
    emission = torch.full((len(path), len(d)), -10.0)
    for i, c in enumerate(path):
        emission[i, d[c]] = 0.0
    times = word_times(
        emission, 16000, ["Hi,", "yo", "42"], MMS_FA.get_tokenizer(), MMS_FA.get_aligner()
    )
    assert times[0] == pytest.approx((0.2, 0.4)) and times[1] == pytest.approx((0.6, 0.8))
    assert times[2] is None  # no letters: wildcard, keeps Whisper's time


def test_refine_trusts_aligned_times_unless_a_word_is_too_long():
    # a word seconds from Whisper's time is kept: around a pause Whisper's can be 5 s off
    aligned = [(0.1, 0.4), (5.0, 5.4), (6.0, 9.0), None]
    assert refine(aligned) == [(0.1, 0.4), (5.0, 5.4), None, None]


class FakeAligner:
    def __init__(self):
        self.calls = []

    def align(self, audio, words):
        self.calls.append((len(audio), words))
        # words 0.4 s long, every 0.5 s from 0.5 s into the clip
        return [(0.3 + i * 0.5 + 0.2, 0.3 + i * 0.5 + 0.6) for i in range(len(words))]


class FakeTranscriber:
    def alternatives(self, audio, spans, language=None):
        self.spans = spans
        return [["think"] for _ in spans]

    def word_rank(self, word):
        return 0


def test_suggest_aligns_segments_with_unsure_words(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: np.zeros(10 * 16000, np.float32))
    (tmp_path / "a.wav").write_bytes(b"x")
    words = [Word(1.0, 1.5, "I", 0.9), Word(1.5, 2.0, "sink", 0.3)]
    t = Transcript(
        "en",
        10,
        [
            Segment(1, 2, "I sink", words=words),
            Segment(5, 6, "fine", words=[Word(5, 6, "fine", 0.99)]),  # nothing unsure: skipped
        ],
    )
    aligner, tr = FakeAligner(), FakeTranscriber()
    with Store(tmp_path / "t.db") as store:
        store.save(tmp_path / "a.wav", "h", t)
        wid = store.segments(1, words=True)[0].words[1].id
        store.correct_word(wid, "think")  # its notebook entry should get the new times too
        assert store.pending_unsure(align=True) == {1: 1}
        assert ops.suggest_words(store, tr, 1, aligner=aligner) == 1
        # aligned on the segment +-1.5 s (from 0, the start), all its words, and offset back
        assert aligner.calls == [(int(3.5 * 16000), ["I", "think"])]
        got = [(w.start, w.end) for w in store.segments(1, words=True)[0].words]
        assert got == [pytest.approx((0.5, 0.9)), pytest.approx((1.0, 1.4))]
        assert tr.spans == [pytest.approx((0.0, 0.4))]  # candidates use the aligned span
        assert store.corrections()[0]["start"] == pytest.approx(1.0)
        assert store.pending_unsure(align=True) == {}
        assert ops.suggest_words(store, tr, 1, aligner=aligner) == 0


def test_word_clip_aligns_its_sentence_first(tmp_path, monkeypatch):
    import io
    import wave

    from recorder.web import App

    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: np.zeros(10 * 16000, np.float32))
    (tmp_path / "a.wav").write_bytes(b"x")
    words = [Word(1.0, 1.5, "I", 0.9), Word(1.5, 2.0, "sink", 0.3)]
    t = Transcript("en", 10, [Segment(1, 2, "I sink", words=words)])
    with Store(tmp_path / "t.db") as store:
        store.save(tmp_path / "a.wav", "h", t)
        wid = store.segments(1, words=True)[0].words[1].id
    aligner = FakeAligner()
    app = App(tmp_path / "t.db", aligner=aligner)
    with wave.open(io.BytesIO(app.word_clip(wid, pad=0))) as f:
        # the aligned 1.0-1.4 s, not Whisper's 1.5-2.0 s
        assert abs(f.getnframes() - 0.4 * 16000) <= 1
    with Store(tmp_path / "t.db") as store:
        assert store.segment_aligned(store.word_context(wid)["segment_id"])
    app.word_clip(wid)
    assert len(aligner.calls) == 1  # stored: not aligned again
