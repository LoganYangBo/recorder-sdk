from recorder import FolderSource, Pipeline, Segment, Store, Transcript


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        return Transcript(
            language="en",
            duration=3.0,
            segments=[Segment(0.0, 1.5, "hello there"), Segment(1.5, 3.0, "how are you")],
        )


class MeLabeler:
    def label(self, path, transcript):
        for s in transcript.segments:
            s.speaker = "me"
        return transcript


def test_process_file_stores_segments(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake audio")
    with Store(tmp_path / "t.db") as store:
        r = Pipeline(FakeTranscriber(), store, MeLabeler()).process_file(audio)
        assert not r.skipped
        segs = store.segments(r.recording_id)
        assert [s.text for s in segs] == ["hello there", "how are you"]
        assert all(s.speaker == "me" for s in segs)
        assert store.search("how are")[0][1].text == "how are you"


def test_same_content_is_skipped(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.wav").write_bytes(b"x")
    (tmp_path / "in" / "copy.m4a").write_bytes(b"x")
    (tmp_path / "in" / "notes.txt").write_text("ignored")
    fake = FakeTranscriber()
    with Store(tmp_path / "t.db") as store:
        results = Pipeline(fake, store).process_source(FolderSource(tmp_path / "in"))
    assert [r.skipped for r in results] == [False, True]
    assert fake.calls == 1
