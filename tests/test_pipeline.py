import sqlite3

from recorder import FolderSource, Pipeline, Segment, Store, Transcript, Word
from recorder.store import MIGRATIONS, SCHEMA_VERSION


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


def test_words_and_confidence_round_trip(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    t = Transcript(
        language="en",
        duration=1.0,
        segments=[
            Segment(
                0.0,
                1.0,
                "hi Logan",
                avg_logprob=-0.2,
                no_speech_prob=0.01,
                words=[Word(0.0, 0.3, "hi", 0.99), Word(0.3, 1.0, "Logan", 0.42)],
            )
        ],
        params={"model": "large-v3", "hotwords": "Logan"},
    )
    with Store(tmp_path / "t.db") as store:
        rid = store.save(audio, "h", t)
        (seg,) = store.segments(rid, words=True)
        assert seg.avg_logprob == -0.2 and seg.no_speech_prob == 0.01
        assert [(w.word, w.probability) for w in seg.words] == [("hi", 0.99), ("Logan", 0.42)]
        assert store.segments(rid)[0].words == []
        assert store.recordings()[0].params == {"model": "large-v3", "hotwords": "Logan"}


def test_reprocess_overwrites_in_place(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.wav").write_bytes(b"x")
    (tmp_path / "in" / "copy.wav").write_bytes(b"x")
    with Store(tmp_path / "t.db") as store:
        (first,) = [
            r
            for r in Pipeline(FakeTranscriber(), store).process_source(
                FolderSource(tmp_path / "in")
            )
            if not r.skipped
        ]
        fake = FakeTranscriber()
        results = Pipeline(fake, store, MeLabeler(), reprocess=True).process_source(
            FolderSource(tmp_path / "in")
        )
        assert fake.calls == 1  # the duplicate copy is not redone
        assert [r.recording_id for r in results] == [first.recording_id] * 2
        assert len(store.recordings()) == 1
        segs = store.segments(first.recording_id)
        assert len(segs) == 2 and all(s.speaker == "me" for s in segs)


def test_migrates_pre_versioning_database(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(MIGRATIONS[0])  # schema as shipped before versioning, user_version 0
    conn.execute(
        "INSERT INTO recordings (path, sha256, duration, language) VALUES ('a.wav', 'h', 1.0, 'en')"
    )
    conn.execute("INSERT INTO segments (recording_id, start, end, text) VALUES (1, 0, 1, 'old')")
    conn.commit()
    conn.close()
    with Store(db) as store:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert store.recordings()[0].params == {}
        assert store.segments(1, words=True)[0].text == "old"
    with Store(db):  # reopening is a no-op
        pass


def test_concurrent_opens_migrate_once(tmp_path):
    import threading

    db = tmp_path / "race.db"
    errors = []

    def open_store():
        try:
            Store(db).close()
        except Exception as e:  # pragma: no cover - the failure being tested for
            errors.append(e)

    threads = [threading.Thread(target=open_store) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with Store(db) as store:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_delete_removes_segments_and_words(tmp_path):
    t = Transcript("en", 1, [Segment(0, 1, "hi", words=[Word(0, 1, "hi", 0.9)])])
    with Store(tmp_path / "t.db") as store:
        rid = store.save(tmp_path / "a.wav", "h", t)
        assert store.delete(rid) and not store.delete(rid)
        counts = [
            store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("recordings", "segments", "words")
        ]
        assert counts == [0, 0, 0]
