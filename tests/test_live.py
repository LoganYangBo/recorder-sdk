import numpy as np

from recorder import Segment, Store, Transcript
from recorder.live import LiveSession

SR = 16000


class BlockTranscriber:
    """Fake Whisper: one segment per full 2 s block, text = the block's tone in Hz."""

    params = {"model": "fake"}

    def __init__(self):
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio) / SR)
        segs = []
        for k in range(len(audio) // (2 * SR)):
            block = audio[k * 2 * SR : (k + 1) * 2 * SR]
            if np.abs(block).max() < 0.01:
                continue  # silence: no segment, like VAD
            hz = np.argmax(np.abs(np.fft.rfft(block))) / 2  # 2 s window -> 0.5 Hz bins
            segs.append(Segment(k * 2.0, k * 2.0 + 2, f"{int(round(hz))}"))
        return Transcript("en", len(audio) / SR, segs)


def tone(hz, seconds=2.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_commits_each_segment_once(tmp_path):
    tr = BlockTranscriber()
    live = LiveSession(tr, settle=1.5, min_new=1.0)
    stream = np.concatenate([tone(100), tone(200), tone(300), tone(400)])
    for i in range(0, len(stream), SR // 2):  # half-second chunks
        live.feed(stream[i : i + SR // 2])
    assert [s.text for s in live.committed] == ["100", "200", "300"]  # last one not settled
    assert live.committed_until == 6.0
    assert max(tr.calls) <= 4.0  # only the uncommitted tail is re-transcribed
    live.finish()
    assert [(s.text, s.start) for s in live.committed] == [
        ("100", 0.0),
        ("200", 2.0),
        ("300", 4.0),
        ("400", 6.0),
    ]

    with Store(tmp_path / "t.db") as store:
        rid = live.save(store, tmp_path / "lib", tr.params)
        rec = store.recording(rid)
        assert rec.path.endswith(".wav") and rec.duration == 8.0
        assert rec.params == {"model": "fake", "live": True} and rec.started_at
        assert [s.text for s in store.segments(rid)] == ["100", "200", "300", "400"]


def test_long_silence_does_not_grow_the_tail():
    tr = BlockTranscriber()
    live = LiveSession(tr, max_tail=6.0)
    for _ in range(20):
        live.feed(np.zeros(SR, np.float32))
    assert live.committed == [] and max(tr.calls) <= 8.0


def test_labels_committed_segments():
    class Labeler:
        def label_audio(self, audio, transcript):
            for s in transcript.segments:
                s.speaker = "me"

    live = LiveSession(BlockTranscriber(), Labeler())
    live.feed(np.concatenate([tone(100), tone(200), tone(300)]))
    live.finish()
    assert [s.speaker for s in live.committed] == ["me"] * 3
