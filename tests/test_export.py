from recorder import Segment, Store, Transcript
from recorder.export import export


def test_export_formats(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    segs = [
        Segment(0, 1.5, "Hi Mia.", "me"),
        Segment(1.6, 3.25, "Shall we go?", "me"),
        Segment(4, 5, "Sure.", "Mia"),
    ]
    with Store(tmp_path / "t.db") as store:
        rid = store.save(
            tmp_path / "a.wav", "h", Transcript("en", 65, segs), started_at="2026-10-16T09:15:00"
        )
        rec, segs = store.recording(rid), store.segments(rid)
    md = export(rec, segs, "md")
    assert md.startswith("# a.wav\n\n2026-10-16 09:15 · 00:01:05 long · en")
    assert "**me** `09:15:00`  \nHi Mia. Shall we go?" in md and "**Mia** `09:15:04`" in md
    srt = export(rec, segs, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n[me] Hi Mia.\n")
    assert "00:00:01,600 --> 00:00:03,250" in srt
    assert export(rec, segs, "txt") == "me: Hi Mia. Shall we go?\nMia: Sure.\n"


def test_cli_export(tmp_path, capsys):
    from recorder import cli

    (tmp_path / "a.wav").write_bytes(b"x")
    with Store(tmp_path / "t.db") as store:
        store.save(tmp_path / "a.wav", "h", Transcript("en", 2, [Segment(0, 2, "hello")]))
    db = str(tmp_path / "t.db")
    cli.main(["--db", db, "export", "1", "-f", "srt", "-o", str(tmp_path / "a.srt")])
    assert (tmp_path / "a.srt").read_text().endswith("hello\n")
    capsys.readouterr()
    cli.main(["--db", db, "export", "1", "-f", "txt"])
    assert capsys.readouterr().out == "hello\n"
