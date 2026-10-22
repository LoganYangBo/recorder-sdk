import numpy as np

from recorder import Segment, Store, Transcript
from recorder.live import LiveSession

SR = 16000


class ToneTranscriber:
    """Fake Whisper: one segment per stretch of tone, text = its pitch in Hz."""

    params = {"model": "fake"}

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, prompt=None):
        self.calls.append((len(audio) / SR, language, prompt))
        step = SR // 10
        loud = [np.abs(audio[i : i + step]).max() > 0.01 for i in range(0, len(audio), step)]
        segs, i = [], 0
        while i < len(loud):
            if not loud[i]:
                i += 1
                continue
            j = i
            while j < len(loud) and loud[j]:
                j += 1
            part = audio[i * step : j * step]
            hz = np.argmax(np.abs(np.fft.rfft(part))) * SR / len(part)
            segs.append(Segment(i / 10, j / 10, f"{int(round(hz, -1))}"))
            i = j
        return Transcript("en", len(audio) / SR, segs)


def tone(hz, seconds=1.5):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def quiet(seconds=1.0):
    return (np.random.default_rng(0).standard_normal(int(seconds * SR)) * 0.001).astype(np.float32)


def feed(live, stream, step=SR // 2):
    for i in range(0, len(stream), step):
        live.feed(stream[i : i + step])


def test_cuts_at_pauses_and_commits_each_phrase_once(tmp_path):
    tr = ToneTranscriber()
    live = LiveSession(tr, pause=0.6)
    feed(live, np.concatenate([quiet(), tone(200), quiet(), tone(300), quiet(), tone(400)]))
    assert [s.text for s in live.committed] == ["200", "300"]  # the last one hasn't paused yet
    assert live.partial == "400"
    live.finish()
    assert [(s.text, round(s.start, 1)) for s in live.committed] == [
        ("200", 1.0),
        ("300", 3.5),
        ("400", 6.0),
    ]
    # each phrase went to Whisper whole, with the text before it as context
    commits = [c for c in tr.calls if c[2] is not None]
    assert commits and commits[-1][1] == "en" and "300" in commits[-1][2]

    with Store(tmp_path / "t.db") as store:
        rid = live.save(store, tmp_path / "lib", tr.params)
        rec = store.recording(rid)
        assert rec.path.endswith(".wav") and rec.duration == 7.5
        assert rec.params == {"model": "fake", "live": True} and rec.started_at
        assert [s.text for s in store.segments(rid)] == ["200", "300", "400"]


def test_talk_without_pauses_is_cut_at_the_longest_piece():
    tr = ToneTranscriber()
    live = LiveSession(tr, max_piece=6.0)
    stream = np.concatenate([quiet(0.5)] + [tone(100 * (k + 2), 1.0) for k in range(12)])
    feed(live, stream)
    assert live.committed and live.committed_until <= 12.5
    assert max(c[0] for c in tr.calls) <= 6.5  # never a longer piece than that
    live.finish()
    assert live.committed_until == live.duration


def test_nothing_is_lost_when_there_is_never_a_quiet_moment():
    live = LiveSession(ToneTranscriber(), max_piece=4.0)
    feed(live, np.concatenate([tone(100 * (k + 2), 1.0) for k in range(10)]))
    live.finish()
    segs = live.committed  # the fake makes one segment of each piece: they cover it all
    assert len(segs) > 1 and segs[0].start == 0 and round(segs[-1].end, 1) == 10.0
    assert all(abs(a.end - b.start) < 0.11 for a, b in zip(segs, segs[1:]))


def test_long_silence_does_not_grow_the_piece():
    tr = ToneTranscriber()
    live = LiveSession(tr)
    feed(live, quiet(50), SR)
    assert live.committed == [] and max(c[0] for c in tr.calls) <= 20.0


def test_old_transcribers_still_work():
    class Plain:
        def transcribe(self, audio):
            return Transcript("en", len(audio) / SR, [Segment(0, 1, "hi")])

    live = LiveSession(Plain())
    feed(live, np.concatenate([tone(200), quiet()]))
    assert [s.text for s in live.committed] == ["hi"]


def test_labels_committed_segments():
    class Labeler:
        def label_audio(self, audio, transcript):
            for s in transcript.segments:
                s.speaker = "me"

    live = LiveSession(ToneTranscriber(), Labeler())
    live.feed(np.concatenate([tone(100), quiet(), tone(200), quiet(), tone(300)]))
    live.finish()
    assert [s.speaker for s in live.committed] == ["me"] * 3
