from __future__ import annotations

import argparse
from pathlib import Path

from . import ops
from .pipeline import Pipeline
from .store import Store
from .sync import FolderSource
from .turns import merge_turns


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _voiceprint_labeler(store: Store, args):
    from .speaker import EcapaEmbedder

    try:
        return ops.voiceprint_labeler(store, EcapaEmbedder(device=args.device), args.threshold)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def _parse_indices(text: str) -> list[int]:
    """'4,7,10-12' -> [4, 7, 10, 11, 12]"""
    out = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def cmd_enroll(args) -> None:
    from .speaker import EcapaEmbedder, load_audio

    with Store(args.db) as store:
        try:
            audios = [load_audio(p) for p in args.paths]
            source = f"{len(args.paths)} file(s)"
            if args.recording is not None:
                idx = _parse_indices(args.segments) if args.segments else []
                audios += ops.segment_audio(store, args.recording, idx)
                source += f" + {len(idx)} segment(s) of recording {args.recording}"
            if not audios:
                raise ValueError("give audio files and/or --recording ID --segments N,M")
            ops.enroll_voice(store, EcapaEmbedder(device=args.device), args.name, audios)
        except ValueError as e:
            raise SystemExit(str(e)) from e
    print(f"saved voiceprint '{args.name}' from {source}")


def cmd_speakers(args) -> None:
    with Store(args.db) as store:
        if args.forget:
            ok = store.delete_voiceprint(args.forget)
            print(f"forgot '{args.forget}'" if ok else f"no voiceprint '{args.forget}'")
            return
        for name in store.voiceprints():
            print(name)


def cmd_relabel(args) -> None:
    with Store(args.db) as store:
        labeler = _voiceprint_labeler(store, args)
        if labeler is None:
            raise SystemExit("no voiceprints yet: run `recorder enroll my_voice.m4a` first")
        ids = args.recording_ids or [r.id for r in store.recordings()]
        ops.relabel(store, labeler, ids, args.rescore)
        for rid in ids:
            print(f"recording {rid}: relabelled")


def cmd_transcribe(args) -> None:
    from .transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        initial_prompt=args.prompt,
        hotwords=args.hotwords,
    )
    with Store(args.db) as store:
        labeler = None if args.no_speakers else _voiceprint_labeler(store, args)
        pipe = Pipeline(transcriber, store, labeler, reprocess=args.reprocess)
        for target in args.paths:
            p = Path(target)
            results = pipe.process_source(FolderSource(p)) if p.is_dir() else [pipe.process_file(p)]
            for r in results:
                status = (
                    "skip (already in db)" if r.skipped else f"ok -> recording {r.recording_id}"
                )
                print(f"{r.path}: {status}")


def cmd_ui(args) -> None:
    from .web import serve

    whisper = {"model": args.model, "language": args.language, "hotwords": args.hotwords}
    serve(
        args.db,
        host=args.host,
        port=args.port,
        device=args.device,
        open_browser=not args.no_open,
        library=args.library,
        whisper=whisper,
    )


def cmd_rename(args) -> None:
    with Store(args.db) as store:
        if not store.rename(args.recording_id, args.title):
            raise SystemExit(f"no recording {args.recording_id}")
        print(f"recording {args.recording_id}: {store.recording(args.recording_id).name}")


def cmd_show(args) -> None:
    with Store(args.db) as store:
        segments = store.segments(args.recording_id)
    if args.turns:
        for t in merge_turns(segments):
            who = f"[{t.speaker}] " if t.speaker else ""
            print(f"{_fmt(t.start)}  {who}{t.text}")
        return
    for i, seg in enumerate(segments, 1):
        who = f"[{seg.speaker}] " if seg.speaker else ""
        if args.scores and seg.speaker_score is not None:
            match = "" if seg.speaker_match == seg.speaker else f" ~{seg.speaker_match}"
            who = f"[{seg.speaker}{match} {seg.speaker_score:+.2f}] "
        print(f"{i:>4}  {_fmt(seg.start)}  {who}{seg.text}")


def cmd_list(args) -> None:
    with Store(args.db) as store:
        for r in store.recordings():
            model = r.params.get("model", "?")
            name = f"{r.title}  ({r.path})" if r.title else r.path
            print(f"{r.id:>4}  {_fmt(r.duration)}  {r.language}  {model:<10}  {name}")


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
    t.add_argument(
        "--prompt", default=None, help="initial prompt, e.g. a sentence with your jargon"
    )
    t.add_argument("--hotwords", default=None, help='names/terms to favour, e.g. "Logan nRF52840"')
    t.add_argument(
        "--reprocess", action="store_true", help="re-transcribe files already in the database"
    )
    t.add_argument("--no-speakers", action="store_true", help="skip me/other labelling")
    t.add_argument("--threshold", type=float, default=0.3, help="min voiceprint similarity")
    t.set_defaults(func=cmd_transcribe)

    e = sub.add_parser("enroll", help="save a voiceprint (yours by default)")
    e.add_argument("paths", nargs="*", help="audio files of only this person speaking")
    e.add_argument("--name", default="me", help='who this is; "me" is the wearer')
    e.add_argument("--recording", type=int, help="take audio from this recording's segments...")
    e.add_argument("--segments", help="...numbered as in `show`, e.g. 4,7,10-12")
    e.add_argument("--device", default="auto", help="auto / cuda / cpu")
    e.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("speakers", help="list enrolled voiceprints")
    sp.add_argument("--forget", metavar="NAME", help="delete a voiceprint")
    sp.set_defaults(func=cmd_speakers)

    r = sub.add_parser("relabel", help="redo me/other labels without re-transcribing")
    r.add_argument("recording_ids", nargs="*", type=int, help="default: all recordings")
    r.add_argument("--threshold", type=float, default=0.3, help="min voiceprint similarity")
    r.add_argument("--rescore", action="store_true", help="re-embed the audio (after a new enroll)")
    r.add_argument("--device", default="auto", help="auto / cuda / cpu")
    r.set_defaults(func=cmd_relabel)

    s = sub.add_parser("show", help="print a recording's transcript")
    s.add_argument("recording_id", type=int)
    s.add_argument("--turns", action="store_true", help="merge segments into speaker turns")
    s.add_argument("--scores", action="store_true", help="show voiceprint similarity")
    s.set_defaults(func=cmd_show)

    sub.add_parser("list", help="list recordings").set_defaults(func=cmd_list)

    rn = sub.add_parser("rename", help="set a recording's display name")
    rn.add_argument("recording_id", type=int)
    rn.add_argument("title", help='new name; "" goes back to the file name')
    rn.set_defaults(func=cmd_rename)

    u = sub.add_parser("ui", help="open the web UI in your browser")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--host", default="127.0.0.1", help="keep the default to stay local-only")
    u.add_argument("--device", default="auto", help="auto / cuda / cpu")
    u.add_argument("--model", default="large-v3", help="Whisper model for imports and live")
    u.add_argument("--language", default=None, help="force language, e.g. en / zh")
    u.add_argument("--hotwords", default=None, help='names/terms to favour, e.g. "Logan"')
    u.add_argument(
        "--library", default=None, help="where imports go (default: recordings/ by the db)"
    )
    u.add_argument("--no-open", action="store_true", help="don't open a browser")
    u.set_defaults(func=cmd_ui)

    q = sub.add_parser("search", help="search transcript text")
    q.add_argument("query")
    q.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
