import numpy as np
import pytest

from recorder import Pipeline, Segment, Store, Transcript, VoiceprintLabeler, enroll, merge_turns
from recorder import speaker as speaker_mod
from recorder.cli import main

SR = speaker_mod.SAMPLE_RATE


def tone(freq, seconds, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class BandEmbedder:
    """Fake embedder: energy below / above 500 Hz. 'Me' talks at 200 Hz, others at 800 Hz."""

    name = "fake"

    def __init__(self):
        self.lengths = []

    def embed(self, audio):
        self.lengths.append(len(audio))
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / SR)
        return np.array([spec[freqs < 500].sum(), spec[freqs >= 500].sum()], dtype=np.float32)


def test_enroll_skips_silence():
    emb = BandEmbedder()
    audio = np.concatenate([tone(200, 6), np.zeros(3 * SR, np.float32)])
    vp = enroll(emb, [audio])
    assert len(emb.lengths) == 2  # the silent 3 s chunk was not embedded
    assert np.isclose(np.linalg.norm(vp), 1.0) and vp[0] > 0.99


def test_enroll_needs_enough_audio():
    with pytest.raises(ValueError):
        enroll(BandEmbedder(), [tone(200, 1)])


def test_labels_me_and_other(monkeypatch):
    audio = np.concatenate([tone(200, 3), tone(800, 3), tone(200, 0.5)])
    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: audio)
    emb = BandEmbedder()
    labeler = VoiceprintLabeler(emb, np.array([1.0, 0.0]), threshold=0.5, min_duration=0.5)
    t = Transcript(
        "en",
        6.5,
        [Segment(0, 3, "mine"), Segment(3, 6, "theirs"), Segment(6.0, 6.5, "short me")],
    )
    labeler.label("x.wav", t)
    assert [s.speaker for s in t.segments] == ["me", "other", "me"]
    assert t.segments[0].speaker_score > 0.9 and t.segments[1].speaker_score < 0.1

    labeler.threshold = 1.01  # stricter threshold, no re-embedding
    n = len(emb.lengths)
    labeler.relabel(t)
    assert [s.speaker for s in t.segments] == ["other"] * 3 and len(emb.lengths) == n


def test_short_segments_are_widened(monkeypatch):
    monkeypatch.setattr(speaker_mod, "load_audio", lambda path: tone(200, 10))
    emb = BandEmbedder()
    labeler = VoiceprintLabeler(emb, np.array([1.0, 0.0]), min_duration=2.0)
    labeler.label("x.wav", Transcript("en", 10, [Segment(5.0, 5.2, "uh")]))
    assert emb.lengths == [2 * SR]


def test_merge_turns():
    segs = [
        Segment(0, 1, "Hello", "me"),
        Segment(1.2, 2, "there.", "me"),
        Segment(2.1, 3, "Hi.", "other"),
        Segment(6, 7, "Later.", "other"),  # long pause: new turn
        Segment(7.1, 8, "Ok.", "me"),
    ]
    turns = merge_turns(segs, max_gap=1.5)
    assert [(t.speaker, t.text) for t in turns] == [
        ("me", "Hello there."),
        ("other", "Hi."),
        ("other", "Later."),
        ("me", "Ok."),
    ]
    assert (turns[0].start, turns[0].end) == (0, 2)


def test_relabel_stored_recording(tmp_path, monkeypatch):
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"x")
    monkeypatch.setattr(
        speaker_mod, "load_audio", lambda path: np.concatenate([tone(200, 2), tone(800, 2)])
    )
    with Store(tmp_path / "t.db") as store:
        t = Transcript("en", 4, [Segment(0, 2, "a"), Segment(2, 4, "b")])
        rid = store.save(audio_path, "h", t)
        labeler = VoiceprintLabeler(BandEmbedder(), np.array([1.0, 0.0]), threshold=0.5)
        Pipeline(None, store, labeler).relabel(rid)
        segs = store.segments(rid)
        assert [s.speaker for s in segs] == ["me", "other"]
        assert segs[0].speaker_score > 0.9


def test_cli_relabel_threshold_only(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    with Store(db) as store:
        t = Transcript(
            "en", 2, [Segment(0, 1, "a", speaker_score=0.6), Segment(1, 2, "b", speaker_score=0.2)]
        )
        rid = store.save("a.wav", "h", t)  # audio file need not exist: no re-embedding
        store.save_voiceprint(
            "me", "speechbrain/spkrec-ecapa-voxceleb", np.ones(4, np.float32).tobytes()
        )
    main(["--db", db, "relabel", "--threshold", "0.5"])
    with Store(db) as store:
        assert [s.speaker for s in store.segments(rid)] == ["me", "other"]
    main(["--db", db, "show", str(rid), "--scores"])
    assert "[me +0.60] a" in capsys.readouterr().out


def test_ecapa_embed_shapes():
    import torch

    class FakeClassifier:
        def encode_batch(self, wavs):
            assert wavs.shape == (1, SR)
            return torch.ones(1, 1, 192)

    emb = speaker_mod.EcapaEmbedder()
    emb._model = FakeClassifier()
    assert emb.embed(np.zeros(SR, np.float32)).shape == (192,)
