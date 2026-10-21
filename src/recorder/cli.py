from __future__ import annotations

import argparse
from pathlib import Path

from . import ops
from .pipeline import Pipeline
from .store import Store
from .sync import AudioFile, FolderSource, find_devices
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


BLE_EVERY = 600  # seconds between Bluetooth syncs with --watch


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


class _Files:
    """A source of explicitly named files."""

    def __init__(self, paths):
        self.paths = paths

    def list_files(self):
        from .sync import started_at_of

        for p in self.paths:
            yield AudioFile(p, started_at_of(p))


def _transcriber(args):
    from .transcribe import WhisperTranscriber

    return WhisperTranscriber(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        initial_prompt=args.prompt,
        hotwords=args.hotwords,
    )


def cmd_transcribe(args) -> None:
    transcriber = _transcriber(args)
    aligner = _LazyAligner(args.device)
    failed = 0
    with Store(args.db) as store:
        labeler = None if args.no_speakers else _voiceprint_labeler(store, args)
        pipe = Pipeline(transcriber, store, labeler, reprocess=args.reprocess)
        for target in args.paths:
            p = Path(target)
            source = FolderSource(p) if p.is_dir() else _Files([p])
            for r in pipe.process_source(source):
                if r.error:
                    failed += 1
                    print(f"{r.path}: FAILED {r.error}")
                    continue
                status = (
                    "skip (already in db)" if r.skipped else f"ok -> recording {r.recording_id}"
                )
                print(f"{r.path}: {status}")
                if not args.no_suggest:
                    _suggest(store, transcriber, r.recording_id, aligner())
    if failed:
        raise SystemExit(f"{failed} file(s) failed")


class _LazyAligner:
    """Loads the aligner (a large download the first time) only when needed."""

    def __init__(self, device: str):
        self.device, self.loaded, self.aligner = device, False, None

    def __call__(self):
        if not self.loaded:
            self.aligner, self.loaded = _aligner(self.device), True
        return self.aligner


class _ClockSetter:
    """Sets the clock of each recorder once when it shows up on USB (needs pyserial)."""

    def __init__(self):
        self.seen: set[str] = set()

    def check(self) -> None:
        try:
            from .clock import recorder_ports, set_clock

            ports = set(recorder_ports())
        except ImportError:
            return
        self.seen &= ports  # unplugged: set it again next time
        for port, stamp in set_clock(sorted(ports - self.seen)):
            print(f"recorder clock set to {stamp} ({port})")
            self.seen.add(port)


def cmd_settime(args) -> None:
    try:
        from .clock import set_clock

        done = set_clock(log=print)
    except ImportError:
        raise SystemExit('needs pyserial: pip install "recorder[device]"')
    if not done:
        raise SystemExit("no recorder answered: switch it on, plug it in, close the serial monitor")
    for port, stamp in done:
        print(f"{port}: clock set to {stamp}")


def _ble(args, work):
    """Connect to the nearby recorder, set its clock, and run `work(recorder)`; None if
    no recorder is in range."""
    import asyncio

    try:
        from .ble import BleRecorder, RecorderError, find_recorder, import_bleak

        BleakClient = import_bleak().BleakClient
    except ImportError:
        raise SystemExit('needs bleak: pip install "recorder[device]"')

    async def main():
        dev = await find_recorder()
        if dev is None:
            return None
        async with BleakClient(dev) as client:
            rec = BleRecorder(client)
            await rec.start()
            print(f"recorder clock set to {await rec.set_clock()}")
            return await work(rec)

    try:
        return asyncio.run(main())
    except (RecorderError, OSError) as e:
        print(f"bluetooth: {e}")
        return None


def _battery(status: dict) -> str:
    return f"battery {status.get('pct', '?')}% ({status.get('bat', '?')} V)"


def cmd_device(args) -> None:
    async def work(rec):
        status = await rec.status()
        files = await rec.list_files()
        return status, files

    found = _ble(args, work)
    if found is None:
        raise SystemExit("no recorder in range: switch it on and bring it close")
    status, files = found
    print(f"{status.get('state', '?')}, {_battery(status)}")
    size = sum(n for _, n in files)
    print(f"{len(files)} recording(s) on the card, {size / 1e6:.1f} MB")


def _sync_ble(args, run, store) -> bool:
    """One Bluetooth sync: download new recordings to inbox/ by the db, import them, and
    with --delete remove them from the card. False if no recorder is in range."""
    from .ble import download_new

    inbox = Path(args.db).resolve().parent / "inbox"

    async def fetch(rec):
        print(_battery(await rec.status()))
        return await download_new(rec, inbox, print)

    files = _ble(args, fetch)
    if files is None:
        return False
    results = run(_Files([local for _, local in files]), store) if files else []
    done = {r.path for r in results if r.status in ("new", "known")}
    gone = [(card, local) for card, local in files if local in done]
    if args.delete and gone:

        async def delete(rec):
            for card, local in gone:
                await rec.delete(card)
                local.unlink(missing_ok=True)
            return len(gone)

        n = _ble(args, delete)
        if n:
            print(f"deleted {n} synced recording(s) from the card")
    return True


def cmd_live(args) -> None:
    import queue
    import threading

    import numpy as np

    from .live import LiveSession

    transcriber = _transcriber(args)
    with Store(args.db) as store:
        labeler = None if args.no_speakers else _voiceprint_labeler(store, args)
        session = LiveSession(transcriber, labeler)
        chunks: queue.Queue = queue.Queue()

        def show(segments):
            for seg in segments:
                who = f"[{seg.speaker}] " if seg.speaker else ""
                print(f"{_fmt(seg.start)}  {who}{seg.text}", flush=True)

        def transcribe():
            stop = False
            while not stop:
                parts = [chunks.get()]  # wait for audio, then take all that's queued
                while not chunks.empty():
                    parts.append(chunks.get_nowait())
                stop = parts[-1] is None
                audio = [p for p in parts if p is not None]
                if audio:
                    show(session.feed(np.concatenate(audio).astype(np.float32) / 32768))

        worker = threading.Thread(target=transcribe, daemon=True)
        worker.start()

        async def stream(rec):
            import asyncio

            print(_battery(await rec.status()))
            await rec.live(chunks.put)
            print("listening (Ctrl+C to stop)...", flush=True)
            while rec.client.is_connected:
                await asyncio.sleep(0.5)
            print("recorder disconnected")
            return True

        try:
            if _ble(args, stream) is None:
                raise SystemExit("no recorder in range: switch it on and bring it close")
        except KeyboardInterrupt:
            pass
        chunks.put(None)
        worker.join()
        show(session.finish())
        if args.save and session.duration > 1:
            library = Path(args.db).resolve().parent / "recordings"
            rid = session.save(store, library, {"model": args.model, "source": "ble"})
            print(f"saved as recording {rid}")


def cmd_sync(args) -> None:
    import time

    from .ingest import Syncer

    library = Path(args.library) if args.library else Path(args.db).resolve().parent / "recordings"
    transcriber = None
    aligner = _LazyAligner(args.device)

    def run(source, store):
        nonlocal transcriber
        transcriber = transcriber or _transcriber(args)
        labeler = None if args.no_speakers else _voiceprint_labeler(store, args)
        syncer = Syncer(
            store,
            Pipeline(transcriber, store, labeler),
            library,
            compress=not args.keep_wav,
            delete_source=args.delete,
            after=None
            if args.no_suggest
            else lambda rid: _suggest(store, transcriber, rid, aligner()),
            log=print,
        )
        results = syncer.sync(source, retry=args.retry)
        counts = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        print(f"{getattr(source, 'root', '')}: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
        return results

    if args.ble:
        with Store(args.db) as store:
            if not args.watch:
                if not _sync_ble(args, run, store):
                    raise SystemExit("no recorder in range: switch it on and bring it close")
                return
            print("looking for the recorder over Bluetooth (Ctrl+C to stop)...")
            try:
                while True:
                    synced = _sync_ble(args, run, store)
                    time.sleep(BLE_EVERY if synced else args.interval)
            except KeyboardInterrupt:
                pass
        return

    clocks = _ClockSetter()
    clocks.check()
    with Store(args.db) as store:
        if args.source:
            run(FolderSource(args.source), store)
            return
        if not args.watch:
            devices = find_devices()
            if not devices:
                raise SystemExit("no recorder found: plug it in, or give a folder to sync")
            for d in devices:
                run(d, store)
            return
        print("waiting for the recorder (Ctrl+C to stop)...")
        synced: set[str] = set()  # devices synced since they were plugged in
        try:
            while True:
                clocks.check()
                devices = {d.id: d for d in find_devices()}
                for dev_id, d in devices.items():
                    if dev_id not in synced:
                        print(f"recorder {dev_id} found at {d.drive}")
                        run(d, store)
                        synced.add(dev_id)
                        print("done; you can unplug it")
                synced &= set(devices)  # unplugged: sync again next time
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


def cmd_dates(args) -> None:
    from .sync import started_at_of

    with Store(args.db) as store:
        for r in store.recordings():
            if r.started_at and not args.force:
                continue
            found = started_at_of(r.path) if Path(r.path).exists() else None
            if found:
                store.set_started_at(r.id, found)
            print(f"{r.id:>4}  {found or '?':<19}  {r.name}")


def cmd_status(args) -> None:
    with Store(args.db) as store:
        files = store.files()
        counts = {}
        for f in files:
            counts[f["status"]] = counts.get(f["status"], 0) + 1
        print(", ".join(f"{n} {k}" for k, n in counts.items()) or "nothing synced yet")
        for f in files:
            if f["status"] != "done":
                print(f"  {f['status']:<8} x{f['attempts']}  {f['source']}  {f['error'] or ''}")


def _aligner(device: str):
    """Forced aligner for precise word times, or None if unavailable."""
    try:
        from .align import WordAligner

        return WordAligner(device=device)
    except Exception as e:
        print(f"word alignment off ({type(e).__name__}: {e}); using Whisper's word times")
        return None


def _suggest(store: Store, transcriber, recording_id: int, aligner=None) -> None:
    n = ops.suggest_words(store, transcriber, recording_id, aligner=aligner)
    if n:
        print(f"  recording {recording_id}: {n} unsure word(s) aligned and analysed")


def cmd_suggest(args) -> None:
    from .transcribe import WhisperTranscriber

    transcriber = WhisperTranscriber(model=args.model, device=args.device, language=args.language)
    aligner = _aligner(args.device)
    with Store(args.db) as store:
        ids = args.recording_ids or list(store.pending_unsure(align=aligner is not None))
        for rid in ids:
            _suggest(store, transcriber, rid, aligner)
        if not ids:
            print("no unsure words left to analyse")


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


def cmd_delete(args) -> None:
    with Store(args.db) as store:
        rec = store.recording(args.recording_id)
        if rec is None or not store.delete(args.recording_id):
            raise SystemExit(f"no recording {args.recording_id}")
    print(f"deleted recording {args.recording_id} ({rec.name}); audio file kept: {rec.path}")


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
            when = (r.started_at or "").replace("T", " ")[:16] or "-"
            print(f"{r.id:>4}  {when:<16}  {_fmt(r.duration)}  {r.language}  {model:<10}  {name}")


def cmd_search(args) -> None:
    with Store(args.db) as store:
        since = until = None
        if args.day:
            from .api import _day_range

            since, until = _day_range(args.day)
        for rec, seg in store.search(args.query, args.limit, args.speaker, since, until):
            who = f"[{seg.speaker}] " if seg.speaker else ""
            when = f"{rec.started_at[:10]} " if rec.started_at else ""
            print(f"#{rec.id} {when}{_fmt(seg.start)}  {who}{seg.text}")


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
    t.add_argument(
        "--hotwords", default=None, help='names/terms to favour, e.g. "PyTorch Kubernetes"'
    )
    t.add_argument(
        "--reprocess", action="store_true", help="re-transcribe files already in the database"
    )
    t.add_argument("--no-speakers", action="store_true", help="skip me/other labelling")
    t.add_argument(
        "--no-suggest", action="store_true", help="skip suggesting corrections for unsure words"
    )
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

    sg = sub.add_parser("suggest", help="suggest corrections for unsure words (done on transcribe)")
    sg.add_argument("recording_ids", nargs="*", type=int, help="default: all not analysed yet")
    sg.add_argument("--model", default="large-v3")
    sg.add_argument("--device", default="auto", help="auto / cuda / cpu")
    sg.add_argument("--language", default=None, help="force language, e.g. en / zh")
    sg.set_defaults(func=cmd_suggest)

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

    dl = sub.add_parser("delete", help="delete a recording from the database (keeps the file)")
    dl.add_argument("recording_id", type=int)
    dl.set_defaults(func=cmd_delete)

    u = sub.add_parser("ui", help="open the web UI in your browser")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--host", default="127.0.0.1", help="keep the default to stay local-only")
    u.add_argument("--device", default="auto", help="auto / cuda / cpu")
    u.add_argument("--model", default="large-v3", help="Whisper model for imports and live")
    u.add_argument("--language", default=None, help="force language, e.g. en / zh")
    u.add_argument("--hotwords", default=None, help='names/terms to favour, e.g. "PyTorch"')
    u.add_argument(
        "--library", default=None, help="where imports go (default: recordings/ by the db)"
    )
    u.add_argument("--no-open", action="store_true", help="don't open a browser")
    u.set_defaults(func=cmd_ui)

    sy = sub.add_parser(
        "sync",
        help="copy recordings from the recorder (or a folder) into the library and transcribe",
    )
    sy.add_argument("source", nargs="?", help="a folder; default: the plugged-in recorder")
    sy.add_argument("--ble", action="store_true", help="over Bluetooth instead of USB/card")
    sy.add_argument("--watch", action="store_true", help="keep waiting for the recorder")
    sy.add_argument("--interval", type=float, default=5, help="seconds between checks (--watch)")
    sy.add_argument(
        "--library", default=None, help="where copies go (default: recordings/ by the db)"
    )
    sy.add_argument(
        "--delete", action="store_true", help="delete files from the source once safely synced"
    )
    sy.add_argument("--keep-wav", action="store_true", help="don't compress WAV to Opus")
    sy.add_argument("--retry", action="store_true", help="also retry files that failed 3 times")
    sy.add_argument("--model", default="large-v3")
    sy.add_argument("--device", default="auto", help="auto / cuda / cpu")
    sy.add_argument("--compute-type", default="default")
    sy.add_argument("--language", default=None)
    sy.add_argument("--prompt", default=None)
    sy.add_argument("--hotwords", default=None)
    sy.add_argument("--no-speakers", action="store_true")
    sy.add_argument("--no-suggest", action="store_true")
    sy.add_argument("--threshold", type=float, default=0.3, help="min voiceprint similarity")
    sy.set_defaults(func=cmd_sync)

    lv = sub.add_parser("live", help="transcribe the recorder live over Bluetooth")
    lv.add_argument("--save", action="store_true", help="store the session as a recording")
    lv.add_argument("--model", default="large-v3")
    lv.add_argument("--device", default="auto", help="auto / cuda / cpu")
    lv.add_argument("--compute-type", default="default")
    lv.add_argument("--language", default=None)
    lv.add_argument("--prompt", default=None)
    lv.add_argument("--hotwords", default=None)
    lv.add_argument("--no-speakers", action="store_true")
    lv.add_argument("--threshold", type=float, default=0.3, help="min voiceprint similarity")
    lv.set_defaults(func=cmd_live)

    dv = sub.add_parser("device", help="the recorder's state, battery and files (Bluetooth)")
    dv.set_defaults(func=cmd_device)

    tm = sub.add_parser("settime", help="set the recorder's clock over USB (needs pyserial)")
    tm.set_defaults(func=cmd_settime)

    dt = sub.add_parser("dates", help="fill in start times from file names / metadata")
    dt.add_argument("--force", action="store_true", help="also redo recordings that have one")
    dt.set_defaults(func=cmd_dates)

    st = sub.add_parser("status", help="files synced, and any that failed")
    st.set_defaults(func=cmd_status)

    q = sub.add_parser("search", help="search transcript text")
    q.add_argument("query", help="words that must all appear (any language)")
    q.add_argument("--speaker", help='only this speaker, e.g. "me"')
    q.add_argument("--day", help="only recordings started on this day, e.g. 2026-10-16")
    q.add_argument("--limit", type=int, default=50)
    q.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
