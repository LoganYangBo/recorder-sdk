from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Pipeline
from .store import Store
from .sync import FolderSource


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cmd_transcribe(args) -> None:
    from .transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber(
        model=args.model, device=args.device, compute_type=args.compute_type, language=args.language
    )
    with Store(args.db) as store:
        pipe = Pipeline(transcriber, store)
        for target in args.paths:
            p = Path(target)
            results = pipe.process_source(FolderSource(p)) if p.is_dir() else [pipe.process_file(p)]
            for r in results:
                status = "skip (already in db)" if r.skipped else f"ok -> recording {r.recording_id}"
                print(f"{r.path}: {status}")


def cmd_show(args) -> None:
    with Store(args.db) as store:
        for seg in store.segments(args.recording_id):
            who = f"[{seg.speaker}] " if seg.speaker else ""
            print(f"{_fmt(seg.start)}  {who}{seg.text}")


def cmd_list(args) -> None:
    with Store(args.db) as store:
        for r in store.recordings():
            print(f"{r.id:>4}  {_fmt(r.duration)}  {r.language}  {r.path}")


def cmd_search(args) -> None:
    with Store(args.db) as store:
        for rec, seg in store.search(args.query):
            print(f"#{rec.id} {_fmt(seg.start)}  {seg.text}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="recorder")
    ap.add_argument("--db", default="recorder.db", help="SQLite database path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe audio files or folders")
    t.add_argument("paths", nargs="+")
    t.add_argument("--model", default="large-v3")
    t.add_argument("--device", default="auto", help="auto / cuda / cpu")
    t.add_argument("--compute-type", default="default", help="e.g. float16, int8_float16, int8")
    t.add_argument("--language", default=None, help="force language, e.g. en / zh")
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("show", help="print a recording's transcript")
    s.add_argument("recording_id", type=int)
    s.set_defaults(func=cmd_show)

    sub.add_parser("list", help="list recordings").set_defaults(func=cmd_list)

    q = sub.add_parser("search", help="search transcript text")
    q.add_argument("query")
    q.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
