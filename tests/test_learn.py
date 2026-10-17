import pytest

from recorder import Segment, Store, Transcript, Word, learn

pytest.importorskip("cmudict")


def test_ipa_and_compare():
    assert learn.ipa("think") == "θɪŋk"
    assert learn.ipa("about") == "əbaʊt"  # unstressed AH is a schwa
    assert learn.ipa("qwzx") is None
    c = learn.compare("think", "sink")
    assert (c["target_ipa"], c["heard_ipa"]) == ("θɪŋk", "sɪŋk")
    assert [o for o in c["ops"] if o["op"] != "same"] == [
        {"op": "sub", "target": "θ", "heard": "s"}
    ]
    missing = [o["op"] for o in learn.compare("desk", "deck")["ops"] if o["op"] != "same"]
    assert missing == ["del"]  # the /s/ of desk was dropped
    assert learn.compare("think", "qwzx") == {"target_ipa": "θɪŋk", "heard_ipa": None, "ops": []}


def test_normalize_and_recase():
    assert learn.normalize("Friday.") == "friday" and learn.normalize("don't") == "don't"
    assert learn.recase("think", "Sink,") == "Think,"
    assert learn.recase("think", "sink") == "think"


def test_candidates_whisper_first_then_sound_alikes():
    common = {"think", "sing", "sank", "sick", "zinc", "sinks"}
    rank = lambda w: sorted(common).index(w) if w in common else None  # noqa: E731
    out = learn.candidates("sink", ["think", "Sink", "think", "thing"], rank)
    assert out[:2] == ["think", "thing"]  # whisper's, most frequent first; "sink" itself dropped
    assert len(out) == learn.MAX_CANDIDATES and "sink" not in out
    assert set(out[2:]) <= common
    assert learn.candidates("qwzx", [], rank) == []  # unknown word, no alternatives: nothing


def make_store(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    words = [Word(0, 0.5, "I", 0.9), Word(0.5, 1, "Sink,", 0.2), Word(1, 1.5, "so.", 0.9)]
    t = Transcript("en", 2, [Segment(0, 1.5, "I Sink, so.", speaker="me", words=words)])
    store = Store(tmp_path / "t.db")
    store.save(audio, "h", t)
    return store


def test_correct_word_logs_and_rewrites(tmp_path):
    with make_store(tmp_path) as store:
        wid = store.segments(1, words=True)[0].words[1].id
        c = store.correct_word(wid, "think")
        assert (c["heard"], c["chosen"], c["sentence"]) == ("sink", "think", "I Think, so.")
        assert (c["speaker"], c["recording"], c["start"]) == ("me", "a.wav", 0.5)
        (seg,) = store.segments(1, words=True)
        assert seg.text == "I Think, so."
        assert (seg.words[1].word, seg.words[1].original) == ("Think,", "Sink,")

        # choosing again keeps the original heard word and replaces the entry
        c = store.correct_word(wid, "thing")
        assert (c["heard"], c["chosen"]) == ("sink", "thing")
        assert [x["chosen"] for x in store.corrections()] == ["thing"]

        # confirming the original puts it back
        store.correct_word(wid, "sink")
        assert store.segments(1)[0].text == "I Sink, so."
        assert [(x["heard"], x["chosen"]) for x in store.corrections()] == [("sink", "sink")]

        # corrections outlive the recording
        store.delete(1)
        assert store.corrections()[0]["recording"] is None
        with pytest.raises(KeyError):
            store.correct_word(wid, "x")
