import numpy as np

from recorder import Recorder, Segment, Store, Transcript, Word, vocab


def test_hotwords_keep_the_newest_that_fit():
    assert vocab.hotwords(["Mia", "su7", "PyTorch"]) == "Mia su7 PyTorch"
    assert vocab.hotwords(["Mia", "su7", "PyTorch"], max_chars=8) == "Mia su7"
    assert vocab.hotwords([]) is None


def test_worth_keeping_names_not_everyday_words():
    assert vocab.worth_keeping("Mia") and vocab.worth_keeping("su7")
    assert vocab.worth_keeping("Model Y") and vocab.worth_keeping("kubectl")
    assert not vocab.worth_keeping("think") and not vocab.worth_keeping("")


def test_candidates_from_a_document():
    text = (
        "Zhang Jie wants to buy a su7. He met Mia at the Xiaomi store in Columbus.\n"
        "We use PyTorch and kubectl; the API is on GCP. The car is fast."
    )
    found = vocab.candidates(text, known={"mia"})
    assert "Jie" in found or "Zhang Jie" in found
    for term in ["su7", "Xiaomi", "Columbus", "PyTorch", "kubectl", "API", "GCP"]:
        assert term in found, term
    assert "Mia" not in found  # already known
    assert not {"He", "We", "The", "car", "fast"} & set(found)


def test_vocab_store_and_sources(tmp_path):
    with Store(tmp_path / "t.db") as store:
        assert store.add_vocab("  PyTorch ") and not store.add_vocab("pytorch")
        store.save_voiceprint("Mia", "m", np.zeros(2, np.float32).tobytes())
        store.save_voiceprint("me", "m", np.zeros(2, np.float32).tobytes())  # not a name
        terms = {v["term"]: v["source"] for v in store.vocab()}
        assert terms == {"PyTorch": "manual", "Mia": "voiceprint"}
        assert store.remove_vocab("pytorch") and not store.remove_vocab("nope")
    with Recorder(tmp_path / "t.db") as rec:
        assert rec.vocab() == ["Mia"]
    get = vocab.from_store(tmp_path / "t.db")
    assert get() == "Mia"


def _one_word(store, tmp_path, speaker):
    seg = Segment(0, 2, "hi me a", speaker, words=[Word(0, 1, "hi", 0.9), Word(1, 2, "me a", 0.2)])
    (tmp_path / "a.wav").write_bytes(b"x")
    store.save(tmp_path / "a.wav", f"h-{speaker}", Transcript("en", 2, [seg]))
    return store.conn.execute("SELECT MAX(id) FROM words").fetchone()[0]


def test_misheard_corrections_feed_the_vocab(tmp_path):
    with Store(tmp_path / "t.db") as store:
        wid = _one_word(store, tmp_path, "other")
        c = store.correct_word(wid, "Mia")  # someone else's words: Whisper misheard
        assert c["reason"] == "misheard"
        assert [v["term"] for v in store.vocab()] == ["Mia"]
        assert store.vocab()[0]["source"] == "correction"


def test_own_words_reason_is_left_to_the_user(tmp_path):
    with Store(tmp_path / "t.db") as store:
        wid = _one_word(store, tmp_path, "me")
        assert store.correct_word(wid, "think")["reason"] is None
        assert store.correct_word(wid, "think", "mispronounced")["reason"] == "mispronounced"
        assert store.vocab() == []
        assert store.correct_word(wid, "me a")["reason"] is None  # confirmed
    with Recorder(tmp_path / "t.db") as rec:
        (w,) = rec.unsure_words(include_reviewed=True)
        assert w.reason is None and rec.corrections()[0].reason is None


def test_cli_vocab(tmp_path, capsys):
    from recorder import cli

    db = str(tmp_path / "t.db")
    notes = tmp_path / "notes.md"
    notes.write_text("Meeting with Mia about the su7 launch.")
    cli.main(["--db", db, "vocab", "add", "PyTorch", "--kind", "term"])
    cli.main(["--db", db, "vocab", "import", str(notes)])
    assert "2 term(s) added: Mia, su7" in capsys.readouterr().out
    cli.main(["--db", db, "vocab"])
    out = capsys.readouterr().out
    assert "PyTorch (term)  [manual" in out and "su7  [import" in out
    cli.main(["--db", db, "vocab", "remove", "su7"])
    assert "removed: su7" in capsys.readouterr().out
