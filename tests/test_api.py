from datetime import date, datetime

from recorder import Recorder, Segment, Store, Transcript, Word


def build(tmp_path):
    db = tmp_path / "t.db"
    with Store(db) as store:
        for i, (started, segs) in enumerate(
            [
                (
                    "2026-10-16T09:00:00",
                    [
                        Segment(
                            0,
                            2,
                            "I think we should",
                            "me",
                            words=[Word(0, 1, "I", 0.9), Word(1, 2, "think", 0.3)],
                        ),
                        Segment(2, 4, "ship it on Friday", "me"),
                        Segment(4, 6, "Sounds good to me", "Alice"),
                    ],
                ),
                (
                    "2026-10-16T23:59:58",
                    [Segment(0, 1, "late night", "me"), Segment(3, 4, "after midnight", "me")],
                ),
                ("2026-10-17T10:00:00", [Segment(0, 2, "我们明天再讨论", "other")]),
                (None, [Segment(0, 1, "no idea when", "me")]),
            ]
        ):
            (tmp_path / f"{i}.wav").write_bytes(b"x")
            store.save(
                tmp_path / f"{i}.wav", f"h{i}", Transcript("en", 10, segs), started_at=started
            )
        store.set_candidates(2, ["sink", "thing"])
    return db


def test_utterances_by_day_and_speaker(tmp_path):
    with Recorder(build(tmp_path)) as rec:
        day = rec.utterances(day="2026-10-16")
        assert [(u.speaker, u.text) for u in day] == [
            ("me", "I think we should ship it on Friday"),  # merged turn
            ("Alice", "Sounds good to me"),
            ("me", "late night"),  # "after midnight" belongs to the 17th
        ]
        assert day[1].at == datetime(2026, 10, 16, 9, 0, 4)
        assert [u.text for u in rec.my_utterances(day=date(2026, 10, 17))] == ["after midnight"]
        assert len(rec.utterances()) == 6 and rec.utterances()[-1].text == "no idea when"
        assert [u.text for u in rec.my_utterances(recording=4)] == ["no idea when"]
        assert [r.id for r in rec.recordings(day="2026-10-16")] == [1, 2]
        (d16, d17) = rec.days()
        assert (d16.date, d16.recordings, d16.speakers) == (
            date(2026, 10, 16),
            2,
            {"me": 4, "Alice": 1},
        )


def test_search(tmp_path):
    with Recorder(build(tmp_path)) as rec:
        assert [u.text for u in rec.search("friday SHIP")] == ["ship it on Friday"]
        assert [u.text for u in rec.search("明天")] == [
            "我们明天再讨论"
        ]  # 2 chars: substring fallback
        assert [u.text for u in rec.search("明天再讨论")] == ["我们明天再讨论"]
        assert [u.text for u in rec.search("me", speaker="Alice")] == ["Sounds good to me"]
        assert rec.search("night", day="2026-10-17") == []
        assert rec.search("night")[0].at == datetime(2026, 10, 16, 23, 59, 58)


def test_unsure_words_and_corrections(tmp_path):
    db = build(tmp_path)
    with Recorder(db) as rec:
        (w,) = rec.unsure_words(speaker="me", day="2026-10-16")
        assert (w.word, w.candidates, w.sentence, w.at) == (
            "think",
            ["sink", "thing"],
            "I think we should",
            datetime(2026, 10, 16, 9, 0, 1),
        )
        rec.store.correct_word(w.word_id, "think")  # confirmed
        assert rec.unsure_words() == []
        assert rec.unsure_words(include_reviewed=True)[0].corrected_from == "think"
        (c,) = rec.corrections()
        assert c.confirmed and rec.corrections(include_confirmed=False) == []


def test_conversations_span_recordings(tmp_path):
    with Recorder(build(tmp_path)) as rec:
        convs = rec.conversations(day="2026-10-16")
        assert len(convs) == 2  # 09:00 and the late-night one, hours apart
        morning = convs[0]
        assert morning.start == datetime(2026, 10, 16, 9, 0, 0)
        assert morning.end == datetime(2026, 10, 16, 9, 0, 6)
        assert morning.recordings == [1] and morning.speakers == {"me": 4.0, "Alice": 2.0}
        assert morning.text.startswith("me: I think we should")
        # the late-night talk runs on past midnight into the next file's day
        late = rec.conversations(since="2026-10-16T23:00:00", until="2026-10-17T01:00:00")
        assert [u.text for u in late[0].utterances] == ["late night", "after midnight"]
        assert len(rec.conversations(day="2026-10-16", min_seconds=5)) == 1


def test_changes_since_a_cursor(tmp_path):
    db = build(tmp_path)
    with Recorder(db) as rec:
        first = rec.changes()
        assert first.changed == [1, 2, 3, 4] and first.deleted == []
        assert rec.changes(first.cursor).changed == []
        rec.store.rename(2, "Late")
        rec.store.correct_word(2, "thing")  # a word of recording 1
        rec.store.delete(3)
        c = rec.changes(first.cursor)
        assert c.changed == [2, 1] and c.deleted == [3] and c.cursor > first.cursor
        assert rec.changes(c.cursor) == type(c)([], [], c.cursor)
